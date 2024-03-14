#include "NanoRenderer.h"
#include <nanovdb/NanoVDB.h>
#include <owl/helper/cuda.h>
#include "cuda_runtime.h"

#include <nanovdb/util/CudaDeviceBuffer.h>
#include <nanovdb/util/Primitives.h>

#include <nanovdb/util/IO.h>

#include <filesystem>


#include "SharedStructs.h"
#include "owl/owl_host.h"

#include <format>

#include "DebugDrawListBase.h"

#include <imgui.h>
#include <imgui_internal.h>

#include <NanoCutterParser.h>

#include <OwlExtensions.h>

#include "FoveatedRendering.h"


#define FOVEATED


extern "C" char NanoRenderer_ptx[];
extern "C" uint8_t NanoRenderer_optixir[];
extern "C" uint32_t NanoRenderer_optixir_length;


using namespace b3d::renderer;
using namespace b3d::renderer::nano;
using namespace owl::common;


namespace
{
	struct GuiData
	{
		bool fillBox{ false };
		std::array<float, 3> fillColor{ 0.8f, 0.3f, 0.2f };

		std::array<float, 2> sampleRemapping{ 0.0f,0.1f };

		std::array<float, 2> fovealPoint{ 0.0f,0.0f };

		SampleIntegrationMethod sampleIntegrationMethode{ SampleIntegrationMethod::maximumIntensityProjection };
	};

	GuiData guiData{};

	struct NanoVdbVolumeDeleter
	{
		auto operator()(const NanoVdbVolume* volume) const noexcept -> void
		{
			OWL_CUDA_CHECK(cudaFree(reinterpret_cast<void*>(volume->grid)));
			delete volume;
		}
	};

	using unique_volume_ptr = std::unique_ptr<NanoVdbVolume, NanoVdbVolumeDeleter>;

	unique_volume_ptr nanoVdbVolume;

	auto createVolume(const std::filesystem::path& file) -> NanoVdbVolume
	{
		//TODO: Let's use shared parameters to grab an initial volume path from the viewer
		// const auto testFile = std::filesystem::path{ "D:/datacubes/n4565_cut/funny.nvdb" };
		const auto testFile =
		std::filesystem::path{ "D:/datacubes/n4565_cut/filtered_level_0_224_257_177_id_7_upscale.fits.nvdb" };
		//std::filesystem::path{ "C:/Users/anton/Downloads/chameleon_1024x1024x1080_uint16.nvdb" };
		//std::filesystem::path{ "C:/Users/anton/Downloads/carp_256x256x512_uint16.nvdb" };
		//const auto testFile = std::filesystem::path{ "D:/datacubes/n4565_cut/nano_level_0_224_257_177.nvdb" };
		// const auto testFile = std::filesystem::path{ "D:/datacubes/ska/40gb/sky_ldev_v2.nvdb" };


		assert(std::filesystem::exists(testFile));
		// owlInstanceGroupSetTransform
		auto volume = NanoVdbVolume{};
		// auto gridVolume = nanovdb::createFogVolumeTorus();
		const auto gridVolume = nanovdb::io::readGrid(testFile.string());
		OWL_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&volume.grid), gridVolume.size()));
		OWL_CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(volume.grid), gridVolume.data(), gridVolume.size(),
			cudaMemcpyHostToDevice));

		const auto gridHandle = gridVolume.grid<float>();
		const auto& map = gridHandle->mMap;
		const auto orientation =
			owl::LinearSpace3f{ map.mMatF[0], map.mMatF[1], map.mMatF[2], map.mMatF[3], map.mMatF[4],
								map.mMatF[5], map.mMatF[6], map.mMatF[7], map.mMatF[8] };
		const auto position = vec3f{ 0.0, 0.0, 0.0 };

		volume.transform = AffineSpace3f{ orientation, position };

		{
			const auto& box = gridVolume.gridMetaData()->worldBBox();
			const auto min = owl::vec3f{ static_cast<float>(box.min()[0]), static_cast<float>(box.min()[1]),
										 static_cast<float>(box.min()[2]) };
			const auto max = owl::vec3f{ static_cast<float>(box.max()[0]), static_cast<float>(box.max()[1]),
										 static_cast<float>(box.max()[2]) };
			volume.worldAabb = owl::box3f{ min, max };
		}

		{
			const auto indexBox = gridHandle->indexBBox();
			const auto boundsMin = nanovdb::Coord{ indexBox.min() };
			const auto boundsMax = nanovdb::Coord{ indexBox.max() + nanovdb::Coord(1) };

			const auto min = owl::vec3f{ static_cast<float>(boundsMin[0]), static_cast<float>(boundsMin[1]),
										 static_cast<float>(boundsMin[2]) };
			const auto max = owl::vec3f{ static_cast<float>(boundsMax[0]), static_cast<float>(boundsMax[1]),
										 static_cast<float>(boundsMax[2]) };

			volume.indexBox = owl::box3f{ min, max };
		}

		return volume;
	}

	auto computeStableEpsilon(const float f) -> float
	{
		return abs(f) * static_cast<float>(1. / (1 << 21));
	}

	auto computeStableEpsilon(const vec3f& v) -> float
	{
		return max(max(computeStableEpsilon(v.x), computeStableEpsilon(v.y)), computeStableEpsilon(v.z));
	}

	auto createRayCameraData(const Camera& camera, const Extent& textureExtent) -> RayCameraData
	{
		const auto origin = vec3f{ camera.origin.x, camera.origin.y, camera.origin.z };
		const auto aspect = textureExtent.width / static_cast<float>(textureExtent.height);

		const auto vz = -normalize(camera.at - origin);
		const auto vx = normalize(cross(camera.up, vz));
		const auto vy = normalize(cross(vz, vx));
		const auto focalDistance = length(camera.at - origin);
		const auto minFocalDistance = max(computeStableEpsilon(origin), computeStableEpsilon(vx));

		const auto screenHeight = 2.f * tanf(camera.FoV / 2.f) * max(minFocalDistance, focalDistance);
		const auto vertical = screenHeight * vy;
		const auto horizontal = screenHeight * aspect * vx;
		const auto lowerLeft = -max(minFocalDistance, focalDistance) * vz - 0.5f * vertical - 0.5f * horizontal;

		return { origin, lowerLeft, horizontal, vertical };
	}

	auto orientedBoxToBox(const owl::box3f& box, const LinearSpace3f& orientation) -> owl::box3f
	{
		const auto extent = owl::vec3f{ box.size().x, box.size().y, box.size().z };

		/*
		 *	  p4----p5
		 *	 / |   / |
		 *	p0----p1 |
		 *	|  p6--|-p7
		 *	| /    |/
		 *	p2----p3
		 */
		const auto p0 = orientation * (0.5f * owl::vec3f(-1.0, -1.0, 1.0) * extent);
		const auto p1 = orientation * (0.5f * owl::vec3f(1.0, -1.0, 1.0) * extent);
		const auto p2 = orientation * (0.5f * owl::vec3f(-1.0, -1.0, -1.0) * extent);
		const auto p3 = orientation * (0.5f * owl::vec3f(1.0, -1.0, -1.0) * extent);
		const auto p4 = orientation * (0.5f * owl::vec3f(-1.0, 1.0, 1.0) * extent);
		const auto p5 = orientation * (0.5f * owl::vec3f(1.0, 1.0, 1.0) * extent);
		const auto p6 = orientation * (0.5f * owl::vec3f(-1.0, 1.0, -1.0) * extent);
		const auto p7 = orientation * (0.5f * owl::vec3f(1.0, 1.0, -1.0) * extent);

		auto newBox = owl::box3f{};
		newBox.extend(p0);
		newBox.extend(p1);
		newBox.extend(p2);
		newBox.extend(p3);
		newBox.extend(p4);
		newBox.extend(p5);
		newBox.extend(p6);
		newBox.extend(p7);

		return newBox;
	}

	std::filesystem::path b3dFilePath{};
} // namespace


auto NanoRenderer::prepareGeometry() -> void
{

	const auto context = nanoContext_.context;

	const auto module = owlModuleCreate(context, NanoRenderer_ptx);

	const auto optixirModule = owlModuleCreateFromIR(context, NanoRenderer_optixir, NanoRenderer_optixir_length);

	[[maybe_unused]] const auto volumeGeometryVars =
		std::array{ OWLVarDecl{ "indexBox", OWL_FLOAT3, OWL_OFFSETOF(NanoVdbVolume, indexBox) },
					OWLVarDecl{ "worldAabb", OWL_FLOAT3, OWL_OFFSETOF(NanoVdbVolume, worldAabb) },
					OWLVarDecl{ "indexBox", OWL_AFFINE3F, OWL_OFFSETOF(NanoVdbVolume, transform) },
					OWLVarDecl{ "grid", OWL_BUFFER_POINTER, OWL_OFFSETOF(NanoVdbVolume, grid) } };

	const auto geometryVars =
		std::array{ OWLVarDecl{ "volume", OWL_USER_TYPE(NanoVdbVolume), OWL_OFFSETOF(GeometryData, volume) }

	};

	const auto geometryType =
		owlGeomTypeCreate(context, OWL_GEOM_USER, sizeof(GeometryData), geometryVars.data(), geometryVars.size());



#ifdef FOVEATED
	const auto rayGenerationVars = std::array{
		OWLVarDecl{ "frameBufferSize", OWL_INT2, OWL_OFFSETOF(RayGenerationFoveatedData, frameBufferSize) },
		OWLVarDecl{ "world", OWL_GROUP, OWL_OFFSETOF(RayGenerationFoveatedData, world) },
		OWLVarDecl{ "foveal", OWL_FLOAT2, OWL_OFFSETOF(RayGenerationFoveatedData, foveal) },
	};
	const auto rayGen = owlRayGenCreate(context, optixirModule, "rayGenerationFoveated", sizeof(RayGenerationFoveatedData),
		rayGenerationVars.data(), rayGenerationVars.size());
#else
	const auto rayGenerationVars = std::array{
		OWLVarDecl{ "frameBufferSize", OWL_INT2, OWL_OFFSETOF(RayGenerationData, frameBufferSize) },
		OWLVarDecl{ "world", OWL_GROUP, OWL_OFFSETOF(RayGenerationData, world) },
	};
	const auto rayGen = owlRayGenCreate(context, optixirModule, "rayGeneration", sizeof(RayGenerationData),
		rayGenerationVars.data(), rayGenerationVars.size());
#endif


	nanoContext_.rayGen = rayGen;

	auto geometry = owlGeomCreate(context, geometryType);

	nanoVdbVolume = unique_volume_ptr(new NanoVdbVolume());

	*nanoVdbVolume.get() = createVolume({});

	const auto volumeSize = nanoVdbVolume->indexBox.size();
	const auto longestAxis = std::max({ volumeSize.x, volumeSize.y, volumeSize.z });

	const auto scale = 1.0f / longestAxis;

	renormalizeScale_ = AffineSpace3f::scale(vec3f{ scale, scale, scale });

	const auto geometryGroup = owlUserGeomGroupCreate(context, 1, &geometry);
	nanoContext_.worldGeometryGroup = owlInstanceGroupCreate(context, 1, &geometryGroup, nullptr, nullptr,
		OWL_MATRIX_FORMAT_OWL, OPTIX_BUILD_FLAG_ALLOW_UPDATE);

	owlGeomSetRaw(geometry, "volume", nanoVdbVolume.get());

	owlGeomTypeSetBoundsProg(geometryType, module, "volumeBounds");
	owlBuildPrograms(context);

	owlGeomSetPrimCount(geometry, 1);

	owlGroupBuildAccel(geometryGroup);
	owlGroupBuildAccel(nanoContext_.worldGeometryGroup);

	owlRayGenSetGroup(rayGen, "world", nanoContext_.worldGeometryGroup);



	owlGeomTypeSetIntersectProg(geometryType, 0, optixirModule, "nano_intersection");
	owlGeomTypeSetClosestHit(geometryType, 0, optixirModule, "nano_closestHit");


	const auto missProgramVars =
		std::array{ OWLVarDecl{ "color0", OWL_FLOAT3, OWL_OFFSETOF(MissProgramData, color0) },
					OWLVarDecl{ "color1", OWL_FLOAT3, OWL_OFFSETOF(MissProgramData, color1) } };

	// ----------- create object  ----------------------------
	nanoContext_.missProgram = owlMissProgCreate(context, optixirModule, "miss", sizeof(MissProgramData),
		missProgramVars.data(), missProgramVars.size());

	// ----------- set variables  ----------------------------
	owlMissProgSet3f(nanoContext_.missProgram, "color0", owl3f{ .8f, 0.f, 0.f });
	owlMissProgSet3f(nanoContext_.missProgram, "color1", owl3f{ .8f, .8f, .8f });

	{
		const auto launchParamsVarsWithStruct = std::array{
			OWLVarDecl{ "cameraData", OWL_USER_TYPE(RayCameraData), OWL_OFFSETOF(LaunchParams, cameraData) },
			OWLVarDecl{ "surfacePointer", OWL_USER_TYPE(cudaSurfaceObject_t),
						OWL_OFFSETOF(LaunchParams, surfacePointer) },
			OWLVarDecl{ "bg.color0", OWL_FLOAT4, OWL_OFFSETOF(LaunchParams, bg.color0) },
			OWLVarDecl{ "bg.color1", OWL_FLOAT4, OWL_OFFSETOF(LaunchParams, bg.color1) },
			OWLVarDecl{ "bg.fillBox", OWL_BOOL, OWL_OFFSETOF(LaunchParams, bg.fillBox) },
			OWLVarDecl{ "bg.fillColor", OWL_FLOAT3, OWL_OFFSETOF(LaunchParams, bg.fillColor) },
			OWLVarDecl{ "colormaps", OWL_USER_TYPE(cudaTextureObject_t), OWL_OFFSETOF(LaunchParams, colorMaps) },
			OWLVarDecl{ "coloringInfo.colorMode", OWL_USER_TYPE(ColoringMode),
						OWL_OFFSETOF(LaunchParams, coloringInfo.coloringMode) },
			OWLVarDecl{ "coloringInfo.singleColor", OWL_FLOAT4, OWL_OFFSETOF(LaunchParams, coloringInfo.singleColor) },
			OWLVarDecl{ "coloringInfo.selectedColorMap", OWL_FLOAT,
						OWL_OFFSETOF(LaunchParams, coloringInfo.selectedColorMap) },
			OWLVarDecl{ "transferFunctionTexture", OWL_USER_TYPE(cudaTextureObject_t),
						OWL_OFFSETOF(LaunchParams, transferFunctionTexture) },
						OWLVarDecl{ "sampleRemapping", OWL_FLOAT2, OWL_OFFSETOF(LaunchParams, sampleRemapping)},
			OWLVarDecl{ "sampleIntegrationMethod", OWL_USER_TYPE(SampleIntegrationMethod),
						OWL_OFFSETOF(LaunchParams, sampleIntegrationMethod) },
		};

		nanoContext_.launchParams =
			owlParamsCreate(nanoContext_.context, sizeof(LaunchParams), launchParamsVarsWithStruct.data(),
				launchParamsVarsWithStruct.size());
	}

	owlBuildPrograms(context);
	owlBuildPipeline(context);
	owlBuildSBT(context);
}

auto NanoRenderer::onRender() -> void
{
	gpuTimers_.nextFrame();



	const auto volumeTransform = renderData_->get<VolumeTransform>("volumeTransform");
	trs_ = volumeTransform->worldMatTRS * renormalizeScale_;

	const auto volumeTranslate = AffineSpace3f::translate(-nanoVdbVolume->indexBox.center());
	const auto groupTransform = trs_ * volumeTranslate;
	owlInstanceGroupSetTransform(nanoContext_.worldGeometryGroup, 0, reinterpret_cast<const float*>(&groupTransform));
	owlGroupRefitAccel(nanoContext_.worldGeometryGroup);
	{
		debugDraw().drawBox(trs_.p / 2, trs_.p, nanoVdbVolume->indexBox.size(), owl::vec4f(0.1f, 0.82f, 0.15f, 1.0f),
			trs_.l);

		const auto aabbSize = orientedBoxToBox(nanoVdbVolume->indexBox, volumeTransform->worldMatTRS.l).size();
		debugDraw().drawBox(trs_.p / 2, trs_.p, aabbSize, owl::vec4f(0.9f, 0.4f, 0.2f, 0.4f), renormalizeScale_.l);
	}

	const auto colorMapParams = colorMapFeature_->getParamsData();
	using namespace owl::extensions;
	owlParamsSetRaw(nanoContext_.launchParams, "coloringInfo.colorMode", &colorMapParams.mode);
	owlParamsSet4f(nanoContext_.launchParams, "coloringInfo.singleColor", colorMapParams.uniformColor);

	owlParamsSet1f(nanoContext_.launchParams, "coloringInfo.selectedColorMap", colorMapParams.selectedColorMap);

	owlParamsSetRaw(nanoContext_.launchParams, "sampleIntegrationMethod", &guiData.sampleIntegrationMethode);


	auto renderTargetFeatureParams = renderTargetFeature_->getParamsData();


	

#ifdef FOVEATED
	const auto foveal = owl2f{ guiData.fovealPoint[0], guiData.fovealPoint[1] };
	owlRayGenSet2f(nanoContext_.rayGen, "foveal", foveal);
#endif


	owlBuildSBT(nanoContext_.context);



	owlParamsSetRaw(nanoContext_.launchParams, "colormaps", &colorMapParams.colorMapTexture);
	owlParamsSet2f(nanoContext_.launchParams, "sampleRemapping",
		owl2f{ guiData.sampleRemapping[0], guiData.sampleRemapping[1] });

	auto transferFunctionParams = transferFunctionFeature_->getParamsData();

	owlParamsSetRaw(nanoContext_.launchParams, "transferFunctionTexture",
		&transferFunctionParams.transferFunctionTexture);



	const auto backgroundColorParams = backgroundColorFeature_->getParamsData();
	owlParamsSet4f(nanoContext_.launchParams, "bg.color0", backgroundColorParams.colors[0]);
	owlParamsSet4f(nanoContext_.launchParams, "bg.color1", backgroundColorParams.colors[1]);

	owlParamsSet1b(nanoContext_.launchParams, "bg.fillBox", guiData.fillBox);
	owlParamsSet3f(nanoContext_.launchParams, "bg.fillColor",
		owl3f{ guiData.fillColor[0], guiData.fillColor[1], guiData.fillColor[2] });



	constexpr auto deviceId = 0; //TODO: Research on device id, in multi gpu system it might be tricky
	const auto stream = owlParamsGetCudaStream(nanoContext_.launchParams, deviceId);
	const auto record = gpuTimers_.record("basic owl rt", stream);

	const auto view = renderData_->get<View>("view");
	assert(view->cameras.size() > 0);

	auto rayCameraData = std::array<RayCameraData, 2>{};
	const auto cameraIndices = view->mode == RenderMode::stereo ? std::vector{ 0,1 } : std::vector{ 0 };
	assert(view->mode == RenderMode::stereo ? renderTargetFeatureParams.colorRT.extent.depth > 1: true);

#ifdef FOVEATED
	const auto r = foveatedFeature_->getLpResources()[0];
	const auto fbSize = owl2i{static_cast<int32_t>(renderTargetFeatureParams.colorRT.extent.width),
							   static_cast<int32_t>(renderTargetFeatureParams.colorRT.extent.height)};
	const auto lrSize = owl2i{static_cast<int32_t>(r.surface.width),
							   static_cast<int32_t>(r.surface.height)};
	owlRayGenSet2i(nanoContext_.rayGen, "frameBufferSize", lrSize);

#else
	const auto fbSize = owl2i{static_cast<int32_t>(renderTargetFeatureParams.colorRT.extent.width),
							   static_cast<int32_t>(renderTargetFeatureParams.colorRT.extent.height)};
	owlRayGenSet2i(nanoContext_.rayGen, "frameBufferSize", fbSize);

#endif

	for (const auto cameraIndex : cameraIndices)
	{
		assert(cameraIndex < view->cameras.size());
		const auto& camera = view->cameras[cameraIndex];

		if (camera.directionsAvailable)
		{
			rayCameraData[cameraIndex] = { camera.origin, camera.dir00, camera.dirDu, camera.dirDv };
		}
		else
		{
			rayCameraData[cameraIndex] = createRayCameraData(camera,
				Extent
				{
					static_cast<uint32_t>(fbSize.x),
					static_cast<uint32_t>(fbSize.y),
					1
				}/*renderTargetFeatureParams.colorRT.extent*/);
		}
	}

	record.start();
	for (const auto cameraIndex : cameraIndices)
	{
		assert(cameraIndex < renderTargetFeatureParams.colorRT.surfaces.size());
		owlParamsSetRaw(nanoContext_.launchParams, "cameraData", &rayCameraData[cameraIndex]);

		const auto lpResource = foveatedFeature_->getLpResources()[cameraIndex];

		owlParamsSetRaw(nanoContext_.launchParams, "surfacePointer", &lpResource.surface.surface);
		//owlParamsSetRaw(nanoContext_.launchParams, "surfacePointer", &renderTargetFeatureParams.colorRT.surfaces[cameraIndex]);
		
		owlAsyncLaunch2D(nanoContext_.rayGen, lrSize.x, lrSize.y, nanoContext_.launchParams);
		
		foveatedFeature_->resolve(renderTargetFeatureParams.colorRT.surfaces[cameraIndex], fbSize.x, fbSize.y, stream, foveal.x, foveal.y);
	}
	record.stop();
}

auto NanoRenderer::onInitialize() -> void
{
	RendererBase::onInitialize();

	const auto context = owlContextCreate(nullptr, 1);

	nanoContext_.context = context;

	prepareGeometry();
}

auto NanoRenderer::onDeinitialize() -> void
{
}

auto NanoRenderer::onGui() -> void
{
	const auto volumeTransform = renderData_->get<VolumeTransform>("volumeTransform");

	ImGui::Begin("RT Settings");



	if (ImGui::Button("spawn box"))
	{
		static auto box = owl::box3f{};
	}

	ImGui::SeparatorText("Integration Method");
	ImGui::BeginGroup();
	ImGui::RadioButton("Maximum Intensity Projection", reinterpret_cast<int*>(&guiData.sampleIntegrationMethode), static_cast<int>(SampleIntegrationMethod::maximumIntensityProjection));
	ImGui::RadioButton("Average Intensity Projection", reinterpret_cast<int*>(&guiData.sampleIntegrationMethode), static_cast<int>(SampleIntegrationMethod::averageIntensityProjection));
	ImGui::RadioButton("Intensity Integration", reinterpret_cast<int*>(&guiData.sampleIntegrationMethode), static_cast<int>(SampleIntegrationMethod::transferIntegration));
	ImGui::EndGroup();
	ImGui::Separator();
	ImGui::Text("Hold SPACE to move foveal point with mouse.");
	ImGui::Text(std::format("foveal x:{:.2} y:{:.2}", guiData.fovealPoint[0], guiData.fovealPoint[1]).c_str());
	ImGui::Separator();

	if (ImGui::GetIO().KeysDown[ImGuiKey_Space])
	{
		const auto mousePosition = ImGui::GetMousePos();

		const auto displaySize = ImGui::GetIO().DisplaySize;

		guiData.fovealPoint[0] = mousePosition.x / static_cast<float>(displaySize.x) * 2.0f - 1.0f;
		guiData.fovealPoint[1] = (1.0 - mousePosition.y / static_cast<float>(displaySize.y)) * 2.0f - 1.0f;

	}

	ImGui::DragFloatRange2("Sample Remapping", &guiData.sampleRemapping[0], &guiData.sampleRemapping[1], 0.0001, -1.0f, 1.0f, "%.4f");

	if (ImGui::Button("Reset Model Transform"))
	{
		volumeTransform->worldMatTRS = AffineSpace3f{};
	}
	ImGui::SeparatorText("Data File (.b3d)");
	ImGui::InputText("##source", const_cast<char*>(b3dFilePath.string().c_str()), b3dFilePath.string().size(),
		ImGuiInputTextFlags_ReadOnly);
	ImGui::SameLine();
	if (ImGui::Button("Select"))
	{
		ImGui::OpenPopup("FileSelectDialog");
	}
	ImGui::SameLine();
	if (ImGui::Button("Load"))
	{
		if (std::filesystem::exists(b3dFilePath))
		{
			const auto t = cutterParser::load(b3dFilePath.generic_string());

			const auto nanoFile = t.clusters.front().accelerationStructureRoot.nanoVdbFile;
		}
		else
		{
			ImGui::TextColored(ImVec4{ 0.9f, 0.1f, 0.1f, 1.0f }, "Error: Can't load file!");
		}
	}

	ImGui::Checkbox("Fill Box", &guiData.fillBox);

	if (guiData.fillBox)
	{
		ImGui::ColorEdit3("Fill Color", guiData.fillColor.data());
	}

	ImGui::SeparatorText("Timings");

	const auto timing = gpuTimers_.get("basic owl rt");

	static float values[100] = {};
	static auto valuesOffset = 0;

	values[valuesOffset] = timing;
	valuesOffset = (valuesOffset + 1) % IM_ARRAYSIZE(values);

	{
		auto average = 0.0f;
		for (auto n = 0; n < IM_ARRAYSIZE(values); n++)
		{
			average += values[n];
		}
		average /= static_cast<float>(IM_ARRAYSIZE(values));
		ImGui::SetNextItemWidth(-1);
		ImGui::PlotHistogram("##perfGraph", values, IM_ARRAYSIZE(values), valuesOffset,
			std::format("avg {:3.2f} ms", average).c_str(), 0.0f, 16.0f, ImVec2(0, 200.0f));
	}

	ImGui::Text("%1.3f", timing);

	debugInfo_.gizmoHelper->drawGizmo(volumeTransform->worldMatTRS);

	static auto currentPath = std::filesystem::current_path();
	static auto selectedPath = std::filesystem::path{};

	const auto center = ImGui::GetMainViewport()->GetCenter();
	ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));

	if (ImGui::BeginPopupModal("FileSelectDialog", nullptr, ImGuiWindowFlags_AlwaysAutoResize))
	{
		constexpr auto roots = std::array{ "A:/", "B:/", "C:/", "D:/", "E:/", "F:/", "G:/", "H:/", "I:/" };

		for (auto i = 0; i < roots.size(); i++)
		{
			const auto root = std::filesystem::path{ roots[i] };
			if (is_directory(root))
			{
				ImGui::SameLine();
				if (ImGui::Button(roots[i]))
				{
					currentPath = root;
				}
			}
		}
		if (ImGui::BeginListBox("##dirs", ImVec2(ImGui::GetFontSize() * 40, ImGui::GetFontSize() * 16)))
		{
			if (ImGui::Selectable("...", false))
			{
				currentPath = currentPath.parent_path();
			}
			auto i = 0;
			for (auto& dir : std::filesystem::directory_iterator{ currentPath })
			{
				i++;
				const auto path = dir.path();
				if (is_directory(path))
				{
					if (ImGui::Selectable(dir.path().string().c_str(), false))
					{
						currentPath = path;
					}
				}
				if (path.has_extension() && path.extension() == ".b3d")
				{
					ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.1f, 0.9f, 0.1f, 1.0f));
					if (ImGui::Selectable(dir.path().string().c_str(), dir.path() == selectedPath))
					{
						selectedPath = dir.path();
					}
					ImGui::PopStyleColor();
				}
			}
			ImGui::EndListBox();
		}
		if (ImGui::Button("OK", ImVec2(120, 0)))
		{
			if (!selectedPath.empty() != 0)
			{
				b3dFilePath = selectedPath;
			}
			ImGui::CloseCurrentPopup();
		}
		ImGui::SetItemDefaultFocus();
		ImGui::SameLine();
		if (ImGui::Button("Cancel", ImVec2(120, 0)))
		{
			selectedPath.clear();
			ImGui::CloseCurrentPopup();
		}
		ImGui::EndPopup();
	}
	ImGui::End();
}
