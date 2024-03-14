#pragma once
#include <owl/common.h>

#include "cuda_runtime.h"



__device__ auto logMap(float scaleRatio, const owl::vec2f& coordinate, const owl::vec2f& foveal,
					   const owl::vec2f& resolution) -> owl::vec2f;
__device__ auto inverseLogMap(const float scaleRatio, const owl::vec2f& coordinate, const owl::vec2f& foveal,
							  const owl::vec2f& resolution)
-> owl::vec2f;
