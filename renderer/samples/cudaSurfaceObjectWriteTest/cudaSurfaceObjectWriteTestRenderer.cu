#include "CudaSurfaceObjectWriteTestRenderer.h"

#include "cuda_runtime.h"

#include "device_launch_parameters.h"
#include "cuda/std/cmath"

#include "math_functions.h"

using namespace b3d::renderer;

__global__ auto writeVertexBuffer(cudaSurfaceObject_t surface, unsigned int width, unsigned int height) -> void
{
	// gridDim: This variable is of type dim3 and contains the dimensions of the grid.
	// blockIdx: This variable is of type uint3 and contains the block index within the grid.
	// blockDim: This variable is of type dim3 and contains the dimensions of the block.
	// threadIdx: This variable is of type uint3 and contains the thread index within the block.

	const auto x = min(blockIdx.x * blockDim.x + threadIdx.x, width - 1);
	const auto y = min(blockIdx.y * blockDim.y + threadIdx.y, height - 1);

	auto val = uint32_t{ 0xFFFFFFFF };
	
	if (x + y == 0)
	{
		// printf("Hello from global thread 0\n");
	}
	surf2Dwrite(val, surface, x * sizeof(uint32_t), y);
}

auto CudaSurfaceObjectWriteTestRenderer::onRender() -> void
{
	auto renderTargetFeatureParams = renderTargetFeature_->getParamsData();

	const auto fbSize = dim3{ renderTargetFeatureParams.colorRT.extent.width,
							   renderTargetFeatureParams.colorRT.extent.height,
							};
	auto cudaRet = cudaSuccess;
	// Execute Kernel
	{
		const auto gridDimXAdd = fbSize.x % 32 == 0 ? 0 : 1; 
		const auto gridDimYAdd = fbSize.y % 32 == 0 ? 0 : 1; 
		auto gridDim =
			dim3{ fbSize.x / 32 + gridDimXAdd, fbSize.y / 32 + gridDimYAdd };
		auto blockDim = dim3{ 32, 32 };
		writeVertexBuffer<<<gridDim, blockDim>>>(renderTargetFeatureParams.colorRT.surfaces[0].surface, fbSize.x, fbSize.y);
		cudaDeviceSynchronize();
		cudaRet = cudaGetLastError();
	}
}

auto CudaSurfaceObjectWriteTestRenderer::onInitialize() -> void
{
	
}

auto CudaSurfaceObjectWriteTestRenderer::onDeinitialize() -> void
{
	
}

auto CudaSurfaceObjectWriteTestRenderer::onGui() -> void
{
	
}
