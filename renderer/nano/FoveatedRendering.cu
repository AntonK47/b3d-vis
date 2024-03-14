#include "FoveatedRendering.h"

#include <owl/common.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "owl/helper/cuda.h"

// #include "FoveatedHelper.cuh"
#include <cuda/std/cmath>


#include "nanovdb/NanoVDB.h"
#include "owl/owl_device.h"

using namespace b3d::renderer;


__global__ auto test() -> void
{
	printf("hello kernel!\n");
}

using namespace owl;

template<typename T>
inline __device__ auto length(const owl::vec_t<T, 2>& v) -> T
{
	return owl::common::polymorphic::sqrt(dot(v, v));
}
__device__ auto inverseLogMap(const float scaleRatio, const vec2f& coordinate, const vec2f& foveal,
	const vec2f& resolution) -> vec2f
{

	const auto maxL = max(
		max(length((vec2f(1, 1) - foveal) * resolution),
			length((vec2f(1, -1) - foveal) * resolution)
		),
		max(length((vec2f(-1, 1) - foveal) * resolution),
			length((vec2f(-1, -1) - foveal) * resolution)
		)
	);
	const float L = log(maxL * 0.5);
	const auto pq = coordinate / resolution * 2.0f - 1.0f - foveal;
	const auto lr = pow(log(length(pq * resolution * 0.5f)) / L, 4.0);
	constexpr auto pi2 = nanovdb::pi<float>() * 2.0f;
	const float theta = atan2f(pq.y * resolution.y, pq.x * resolution.x) / pi2 + (pq.y < 0.0f ? 1.0f : 0.0);
	float theta2 = atan2f(pq.y * resolution.y, -abs(pq.x) * resolution.x) / pi2 + (pq.y < 0.0f ? 1.0f : 0.0);

	const auto logCoordinate = vec2f(lr, theta) / scaleRatio;
	return logCoordinate * resolution;
}


struct FovealParameter
{
	vec2f foveal;
	float scaleRatio;
};

__global__ auto resolveLp(cudaTextureObject_t inputTextureObj, cudaSurfaceObject_t outputSurfObj, const int width,
	const int height, const FovealParameter fovealParameter)
	-> void
{
	const auto foveal = fovealParameter.foveal;
	const auto resolution = vec2f(width, height);

	const auto x = blockIdx.x * blockDim.x + threadIdx.x;
	const auto y = blockIdx.y * blockDim.y + threadIdx.y;
	const auto pixelIndex = vec2f(x, y);

	//const auto pixelIndex = vec2f(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
	if (pixelIndex.x >= width || pixelIndex.y >= height)
	{
		return;
	}
	const auto uv = vec2f(pixelIndex) / resolution / fovealParameter.scaleRatio;

	auto newPixelIndex = vec2f(pixelIndex);
	/*if (uv.x > 1.0 || uv.y > 1.0) {
		if (uv.y > 1.0 && uv.y < 1.0 + 1.0 / resolution.y) {
			newPixelIndex -= resolution / fovealParameter.scaleRatio;
		}
		else {
			return;
		}
	}*/

	

	auto screen = (vec2f(newPixelIndex));// + vec2f(.5f)); //*2.0f -1.0f;

	screen = inverseLogMap(fovealParameter.scaleRatio, screen, foveal, resolution);
	screen /= resolution;
	//printf("hello kernel!\n");
	//printf("%f , %f \n", screen.x, screen.y);

	const float4 result = tex2D<float4>(inputTextureObj,screen.x, screen.y);
	//printf("%0.3f , %0.3f , %0.3f , %0.3f \n", result.x, result.y, result.z, result.w);


	
	surf2Dwrite(owl::make_rgba(vec4f(result.x, result.y, result.z, result.w))/*owl::make_rgba(vec4f{result})*/, outputSurfObj, sizeof(uint32_t) * pixelIndex.x, pixelIndex.y);


}

auto init(const uint32_t width, const uint32_t height, const float scaleRation = 2.0f)
{

}

auto b3d::renderer::FoveatedRenderingFeature::onInitialize() -> void
{
	createResources();
}
auto b3d::renderer::FoveatedRenderingFeature::onDeinitialize() -> void
{
	destroyResources();
}
auto b3d::renderer::FoveatedRenderingFeature::gui() -> void
{
}
auto b3d::renderer::FoveatedRenderingFeature::resolve(const CudaSurfaceResource& surface, const uint32_t width, const uint32_t height, const CUstream stream, float fovX, float fovY) -> void
{
	dim3 threadsperBlock(16, 16);
	const auto x = (width + threadsperBlock.x - 1) / threadsperBlock.x;
	const auto y = (height + threadsperBlock.y - 1) / threadsperBlock.y;
	dim3 numBlocks(x, y);


	resolveLp << <numBlocks, threadsperBlock, 0, stream >> > (
		lpResources_.front().texture,
		surface.surface,
		width,
		height,
		{ vec2f{fovX,fovY},resolutionScaleRatio_ });


}
auto b3d::renderer::FoveatedRenderingFeature::destroyResources() -> void
{
	for (const auto& resource : lpResources_)
	{
		OWL_CUDA_CHECK(cudaDestroySurfaceObject(resource.surface.surface));
		OWL_CUDA_CHECK(cudaDestroyTextureObject(resource.texture));
		OWL_CUDA_CHECK(cudaFreeArray(resource.surface.buffer));
	}

	lpResources_.clear();
}
auto b3d::renderer::FoveatedRenderingFeature::createResources() -> void
{
	const auto renderTargets = sharedParameters_->get<RenderTargets>("renderTargets");
	assert(renderTargets);
	inputWidth_ = renderTargets->colorRt.extent.width;
	inputHeight_ = renderTargets->colorRt.extent.height;
	const auto newWidth = static_cast<size_t>(renderTargets->colorRt.extent.width / resolutionScaleRatio_);
	const auto newHeight = static_cast<size_t>(renderTargets->colorRt.extent.height / resolutionScaleRatio_);
	lpWidth_ = newWidth;
	lpHeight_ = newHeight;
	const auto channelDesc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);


	for (auto i = 0; i < renderTargets->colorRt.extent.depth; i++)
	{
		auto lpBuffer = cudaArray_t{};
		OWL_CUDA_CHECK(cudaMallocArray(&lpBuffer, &channelDesc, newWidth, newHeight, cudaArraySurfaceLoadStore));

		const auto resourceInfo = cudaResourceDesc{ cudaResourceTypeArray, { lpBuffer } };

		auto lpSurfaceObject = cudaSurfaceObject_t{};
		OWL_CUDA_CHECK(cudaCreateSurfaceObject(&lpSurfaceObject, &resourceInfo));

		auto textureInfo = cudaTextureDesc{};
		textureInfo.addressMode[0] = cudaAddressModeClamp;
		textureInfo.addressMode[1] = cudaAddressModeClamp;

		textureInfo.filterMode = cudaFilterModeLinear;
		textureInfo.readMode = cudaReadModeNormalizedFloat;

		textureInfo.normalizedCoords = 1;
		textureInfo.maxAnisotropy = 1;
		textureInfo.maxMipmapLevelClamp = 0;
		textureInfo.minMipmapLevelClamp = 0;
		textureInfo.mipmapFilterMode = cudaFilterModePoint;
		textureInfo.borderColor[0] = 1.0f;
		textureInfo.borderColor[1] = 1.0f;
		textureInfo.borderColor[2] = 1.0f;
		textureInfo.borderColor[3] = 1.0f;
		textureInfo.sRGB = 0;
		auto lpTexture = cudaTextureObject_t{};
		OWL_CUDA_CHECK(cudaCreateTextureObject(&lpTexture, &resourceInfo, &textureInfo, nullptr));

		lpResources_.push_back({ { lpBuffer, lpSurfaceObject, newWidth, newHeight }, lpTexture });
	}
}
auto b3d::renderer::FoveatedRenderingFeature::beginUpdate() -> void
{
	const auto renderTargets = sharedParameters_->get<RenderTargets>("renderTargets");
	assert(renderTargets);
	const auto width = renderTargets->colorRt.extent.width;
	const auto height = renderTargets->colorRt.extent.height;
	if (inputWidth_ != width || inputHeight_ != height)
	{
		OWL_CUDA_CHECK(cudaDeviceSynchronize());
		destroyResources();
		createResources();
	}
	assert(inputWidth_ > 0 && inputHeight_ > 0);
}
auto testCall(CUstream stream) -> void
{
	auto gridDim =
		dim3{ 1, 1 };
	auto blockDim = dim3{ 1, 1 };
	test << <gridDim, blockDim >> > ();
}
