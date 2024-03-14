#pragma once

#include <array>
#include <owl/common.h>

#include "cuda_runtime.h"

namespace b3d::renderer
{
	struct Camera
	{
		owl::vec3f origin;
		owl::vec3f at;
		owl::vec3f up;
		float cosFoV;
		float FoV; // in radians
		bool directionsAvailable{ false };
		owl::vec3f dir00;
		owl::vec3f dirDu;
		owl::vec3f dirDv;
	};

	struct Extent
	{
		uint32_t width;
		uint32_t height;
		uint32_t depth;
	};

	struct CudaSurfaceResource
	{
		cudaArray_t buffer;
		cudaSurfaceObject_t surface;
		size_t width{};
		size_t height{};
	};

	struct CudaStereoRenderTarget
	{
		std::array<CudaSurfaceResource, 2> surfaces;
		Extent extent;
	};

	struct ExternalRenderTarget
	{
		cudaGraphicsResource_t target;
		Extent extent;
		void* nativeHandle{ nullptr };
	};

	// ExternalTexture is read only
	using ExternalTexture = ExternalRenderTarget;

	enum class RenderMode : int
	{
		mono = 0,
		stereo
	};

	struct View
	{
		std::array<Camera, 2> cameras;
		RenderMode mode;
	};

	struct RenderTargets
	{
		ExternalRenderTarget colorRt;
		ExternalRenderTarget minMaxRt;
	};

	struct Synchronization
	{
		cudaExternalSemaphore_t waitSemaphore;
		cudaExternalSemaphore_t signalSemaphore;
		uint64_t fenceValue{ 0 };
	};

	struct VolumeTransform
	{
		owl::affine3f worldMatTRS{};
	};

	struct RendererInitializationInfo
	{
		cudaUUID_t deviceUuid;
		int deviceIndex;
	};

	enum class ColoringMode
	{
		single = 0,
		colormap = 1
	};

	struct ColorRGB
	{
		float r, g, b;
	};

	struct ColorRGBA
	{
		float r, g, b, a;
	};

	struct ColoringInfo
	{
		ColoringMode coloringMode;
		owl::vec4f singleColor;
		float selectedColorMap;
		std::array<ColorRGBA, 2> backgroundColors;
	};

	struct ColorMapInfos
	{
		std::vector<std::string>* colorMapNames{};
		int colorMapCount;
		float firstColorMapYTextureCoordinate;
		float colorMapHeightNormalized;
	};

}
