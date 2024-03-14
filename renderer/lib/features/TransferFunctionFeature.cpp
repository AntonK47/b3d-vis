#include "TransferFunctionFeature.h"

#include "Logging.h"

#include "Curve.h"
#include "owl/helper/cuda.h"

#define IMGUI_DEFINE_MATH_OPERATORS
#include "imgui.h"

using namespace b3d::renderer;

TransferFunctionFeature::TransferFunctionFeature(const std::string& name, const size_t dataPointsCount)
	: RenderFeature{ name }, dataPoints_(dataPointsCount), stagingBuffer_(512), transferFunctionTexture_{ nullptr }
{
	assert(dataPointsCount > 0);
	dataPoints_[0].x = ImGui::CurveTerminator;
	newDataAvailable_ = true;
}

auto TransferFunctionFeature::beginUpdate() -> void
{
	transferFunctionTexture_ = sharedParameters_->get<ExternalTexture>("transferFunctionTexture");

	skipUpdate = transferFunctionTexture_ == nullptr;

	if (skipUpdate)
	{
		b3d::renderer::log("TransferFunctionFeature skips update, because of missing shared parameters!");
		return;
	}
	
	OWL_CUDA_CHECK(cudaGraphicsMapResources(1, &transferFunctionTexture_->target));


	OWL_CUDA_CHECK(
		cudaGraphicsSubResourceGetMappedArray(&transferFunctionCudaArray_, transferFunctionTexture_->target, 0, 0));

	if (newDataAvailable_)
	{
		newDataAvailable_ = false;
		if(stagingBuffer_.size() != transferFunctionTexture_->extent.width)
		{
			b3d::renderer::log("TransferFunctionFeature: Size of staging buffer and texture memory don't match!");
		}
		const auto minWidth = std::min<size_t>(stagingBuffer_.size(), transferFunctionTexture_->extent.width);
		const auto minWidthBytes = minWidth * sizeof(float);
		cudaMemcpy2DToArrayAsync(transferFunctionCudaArray_, 0, 0, stagingBuffer_.data(), minWidthBytes, minWidthBytes,
								 1, cudaMemcpyHostToDevice);
	}

	// Create texture
	auto resDesc = cudaResourceDesc{};
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = transferFunctionCudaArray_;
	
	
	auto texDesc = cudaTextureDesc{};
	texDesc.addressMode[0] = cudaAddressModeClamp;
	texDesc.addressMode[1] = cudaAddressModeClamp;

	texDesc.filterMode = cudaFilterModeLinear;
	texDesc.readMode = cudaReadModeElementType; // cudaReadModeNormalizedFloat

	texDesc.normalizedCoords = 1;
	texDesc.maxAnisotropy = 1;
	texDesc.maxMipmapLevelClamp = 0;
	texDesc.minMipmapLevelClamp = 0;
	texDesc.mipmapFilterMode = cudaFilterModePoint;
	texDesc.borderColor[0] = 1.0f;
	texDesc.borderColor[1] = 1.0f;
	texDesc.borderColor[2] = 1.0f;
	texDesc.borderColor[3] = 1.0f;
	texDesc.sRGB = 0;

	OWL_CUDA_CHECK(cudaCreateTextureObject(&transferFunctionCudaTexture_, &resDesc, &texDesc, nullptr));
	
}

auto TransferFunctionFeature::endUpdate() -> void
{
	if (skipUpdate)
	{
		return;
	}
	OWL_CUDA_CHECK(cudaDestroyTextureObject(transferFunctionCudaTexture_));
	OWL_CUDA_CHECK(cudaGraphicsUnmapResources(1, &transferFunctionTexture_->target));

}

auto TransferFunctionFeature::gui() -> void
{
	const auto availableSize = ImGui::GetContentRegionAvail();
	const auto size = ImVec2{ availableSize.x, std::min({ 200.0f, availableSize.y }) };

	if (ImGui::Curve("##transferFunction", size, dataPoints_.size(), dataPoints_.data(), &selectedCurveHandleIdx_))
	{
		// curve changed
		// TODO: maybe trigger an event to refit/recreate data
		b3d::renderer::log("Params changed");
		newDataAvailable_ = true;
	}
	if (newDataAvailable_)
	{
		stagingBuffer_.resize(transferFunctionTexture_->extent.width);

		const auto inc = 1.0f / (stagingBuffer_.size() - 1);
		for (auto i = 0; i < stagingBuffer_.size(); i++)
		{
			stagingBuffer_[i] = ImGui::CurveValue(i * inc, dataPoints_.size(), dataPoints_.data());
		}
	}
}

auto TransferFunctionFeature::getParamsData() -> ParamsData
{
	return { transferFunctionCudaTexture_ };
}
auto TransferFunctionFeature::hasGui() const -> bool
{
	return true;
}
