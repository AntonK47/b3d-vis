#pragma once
#include <RenderFeature.h>
#include <cuda.h>

namespace b3d::renderer
{
	class FoveatedRenderingFeature final : public RenderFeature
	{
	public:
		explicit FoveatedRenderingFeature() : RenderFeature{ "FoveatedRendering" }
		{
		}

		auto onInitialize() -> void override;
		auto onDeinitialize() -> void override;
		auto gui() -> void override;
		[[nodiscard]] inline auto hasGui() const -> bool override
		{
			return true;
		}
		auto resolve(const CudaSurfaceResource& surface, const uint32_t width, const uint32_t height, const CUstream stream, float fovX, float fovY) -> void;

	private:
		auto destroyResources() -> void;
		auto createResources() -> void;

		struct LpResource
		{
			CudaSurfaceResource surface;
			cudaTextureObject_t texture;
		};
	public:
		auto beginUpdate() -> void override;

		[[nodiscard]] inline auto getLpResources() const -> std::vector<LpResource>
		{
			return lpResources_;
		}

	private:
		float resolutionScaleRatio_{ 1.0 };

		std::vector<LpResource> lpResources_{}; // LP - Log Polar

		size_t lpWidth_{};
		size_t lpHeight_{};
		size_t inputWidth_{};
		size_t inputHeight_{};
	};
}

auto testCall(CUstream stream) -> void;
