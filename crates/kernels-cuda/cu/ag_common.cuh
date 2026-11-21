// Common device definitions for AgCAD Tier N kernels.
#pragma once
#include "ag_status.cuh"

typedef unsigned long long u64;
typedef unsigned int u32;
typedef unsigned short u16;
typedef long long i64;

#define AG_TID (blockIdx.x * blockDim.x + threadIdx.x)
#define AG_KERNEL extern "C" __global__

template <typename T> __device__ __forceinline__ T* ag_ptr(u64 a) { return reinterpret_cast<T*>(a); }
