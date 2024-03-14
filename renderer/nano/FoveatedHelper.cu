#include "FoveatedHelper.cuh"
using namespace owl;

#include "nanovdb/NanoVDB.h"

template<typename T>
 inline __device__ auto length(const owl::vec_t<T, 2>& v) -> T
{
	return owl::common::polymorphic::sqrt(dot(v, v));
}




