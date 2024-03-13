#pragma once

#include "RenderFeature.h"

#include <vector>

#include "imgui.h"

namespace b3d::renderer
{
	class TransferFunctionFeature final : public RenderFeature
	{
	public:
		explicit TransferFunctionFeature(const std::string& name, const size_t dataPointsCount = 32);
		auto beginUpdate() -> void override;
		auto endUpdate() -> void override;
		auto gui() -> void override;
		struct ParamsData
		{
			//TODO: put your computed params here
			cudaTextureObject_t transferFunctionTexture{};

		};

		[[nodiscard]] auto getParamsData() -> ParamsData;
		[[nodiscard]] auto hasGui() const -> bool override;

	private:
		bool skipUpdate{ false };

		int selectedCurveHandleIdx_{-1};

		std::vector<ImVec2> dataPoints_;
		std::vector<float> stagingBuffer_;

		bool newDataAvailable_{ false };

		ExternalTexture* transferFunctionTexture_;

		cudaArray_t transferFunctionCudaArray_{ nullptr };
		cudaTextureObject_t transferFunctionCudaTexture_{};
	};
} // namespace b3d::renderer
