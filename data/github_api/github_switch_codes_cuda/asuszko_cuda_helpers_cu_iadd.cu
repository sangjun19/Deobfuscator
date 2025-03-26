// Repository: asuszko/cuda_helpers
// File: src/cu_iadd.cu

#include <cuda.h>

#include "cu_iadd.h"
#include "cu_errchk.h"

#define BLOCKSIZE 128
const int bs = 128;


template <typename T>
__global__ void add1_val(T* __restrict__ y, const T x, unsigned long long N)
{
    unsigned long long index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long stride = gridDim.x * blockDim.x;

    #pragma unroll bs
    for(; index < N; index += stride) {
        y[index] += x;
    }
}


template <typename T>
__global__ void add1_vec(T* __restrict__ y, const T* __restrict__ x, unsigned long long N)
{
    unsigned long long index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long stride = gridDim.x * blockDim.x;

    #pragma unroll bs
    for(; index < N; index += stride) {
        y[index] += x[index];
    }
}


template <typename T>
__global__ void add2_val(T* __restrict__ y, const T x, unsigned long long N)
{
    unsigned long long index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long stride = gridDim.x * blockDim.x;

    #pragma unroll bs
    for(; index < N; index += stride) {
        T valy = y[index];

        valy.x += x.x;
        valy.y += x.y;

        y[index] = valy;
    }
}


template <typename T>
__global__ void add2_vec(T* __restrict__ y, const T* __restrict__ x, unsigned long long N)
{
    unsigned long long index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long stride = gridDim.x * blockDim.x;

    #pragma unroll bs
    for(; index < N; index += stride) {
        T valy = y[index];
        T valx = x[index];

        valy.x += valx.x;
        valy.y += valx.y;

        y[index] = valy;
    }
}


template <typename T>
__global__ void add3_val(T* __restrict__ y, const T x, unsigned long long N)
{
    unsigned long long index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long stride = gridDim.x * blockDim.x;

    #pragma unroll bs
    for(; index < N; index += stride) {
        T valy = y[index];

        valy.x += x.x;
        valy.y += x.y;
        valy.z += x.z;

        y[index] = valy;
    }
}


template <typename T>
__global__ void add3_vec(T* __restrict__ y, const T* __restrict__ x, unsigned long long N)
{
    unsigned long long index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long stride = gridDim.x * blockDim.x;

    #pragma unroll bs
    for(; index < N; index += stride) {
        T valy = y[index];
        T valx = x[index];

        valy.x += valx.x;
        valy.y += valx.y;
        valy.z += valx.z;

        y[index] = valy;
    }
}


template <typename T>
__global__ void add4_val(T* __restrict__ y, const T x, unsigned long long N)
{
    unsigned long long index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long stride = gridDim.x * blockDim.x;

    #pragma unroll bs
    for(; index < N; index += stride) {
        T valy = y[index];

        valy.x += x.x;
        valy.y += x.y;
        valy.z += x.z;
        valy.w += x.w;

        y[index] = valy;
    }
}


template <typename T>
__global__ void add4_vec(T* __restrict__ y, const T* __restrict__ x, unsigned long long N)
{
    unsigned long long index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long stride = gridDim.x * blockDim.x;

    #pragma unroll bs
    for(; index < N; index += stride) {
        T valy = y[index];
        T valx = x[index];

        valy.x += valx.x;
        valy.y += valx.y;
        valy.z += valx.z;
        valy.w += valx.w;

        y[index] = valy;
    }
}



void cu_iadd_vec(void *y, void *x, unsigned long long N,
                 int dtype, int dtype_len,
                 cudaStream_t *stream)
{
    dim3 blockSize(bs);
    dim3 gridSize((((N-1)/blockSize.x+1)-1)/blockSize.x+1);

    cudaStream_t stream_id;
    (stream == NULL) ? stream_id = NULL : stream_id = *stream;

    switch(dtype) {
        case 0:
            switch(dtype_len) {
                case 1:
                    add1_vec<<<gridSize,blockSize,0,stream_id>>>(static_cast<float*>(y),
                                                                 static_cast<const float*>(x), N);
                    break;
                case 2:
                    add2_vec<<<gridSize,blockSize,0,stream_id>>>(static_cast<float2*>(y),
                                                                 static_cast<const float2*>(x), N);
                    break;
                case 3:
                    add3_vec<<<gridSize,blockSize,0,stream_id>>>(static_cast<float3*>(y),
                                                                 static_cast<const float3*>(x), N);
                    break;
                case 4:
                    add4_vec<<<gridSize,blockSize,0,stream_id>>>(static_cast<float4*>(y),
                                                                 static_cast<const float4*>(x), N);
                    break;
            }
            break;
        case 1:
            switch(dtype_len) {
                case 1:
                    add1_vec<<<gridSize,blockSize,0,stream_id>>>(static_cast<double*>(y),
                                                                 static_cast<const double*>(x), N);
                    break;
                case 2:
                    add2_vec<<<gridSize,blockSize,0,stream_id>>>(static_cast<double2*>(y),
                                                                 static_cast<const double2*>(x), N);
                    break;
                case 3:
                    add3_vec<<<gridSize,blockSize,0,stream_id>>>(static_cast<double3*>(y),
                                                                 static_cast<const double3*>(x), N);
                    break;
                case 4:
                    add4_vec<<<gridSize,blockSize,0,stream_id>>>(static_cast<double4*>(y),
                                                                 static_cast<const double4*>(x), N);
                    break;
            }
            break;
    }
}


void cu_iadd_val(void *y, void *x, unsigned long long N,
                 int dtype, int dtype_len,
                 cudaStream_t *stream)
{
    dim3 blockSize(bs);
    dim3 gridSize((((N-1)/blockSize.x+1)-1)/blockSize.x+1);

    cudaStream_t stream_id;
    (stream == NULL) ? stream_id = NULL : stream_id = *stream;

    switch(dtype) {
        case 0:
            switch(dtype_len) {
                case 1:
                    add1_val<<<gridSize,blockSize,0,stream_id>>>(static_cast<float*>(y),
                                                                (static_cast<const float*>(x))[0], N);
                    break;
                case 2:
                    add2_val<<<gridSize,blockSize,0,stream_id>>>(static_cast<float2*>(y),
                                                                (static_cast<const float2*>(x))[0], N);
                    break;
                case 3:
                    add3_val<<<gridSize,blockSize,0,stream_id>>>(static_cast<float3*>(y),
                                                                (static_cast<const float3*>(x))[0], N);
                    break;
                case 4:
                    add4_val<<<gridSize,blockSize,0,stream_id>>>(static_cast<float4*>(y),
                                                                (static_cast<const float4*>(x))[0], N);
                    break;
            }
            break;
        case 1:
            switch(dtype_len) {
                case 1:
                    add1_val<<<gridSize,blockSize,0,stream_id>>>(static_cast<double*>(y),
                                                                (static_cast<const double*>(x))[0], N);
                    break;
                case 2:
                    add2_val<<<gridSize,blockSize,0,stream_id>>>(static_cast<double2*>(y),
                                                                (static_cast<const double2*>(x))[0], N);
                    break;
                case 3:
                    add3_val<<<gridSize,blockSize,0,stream_id>>>(static_cast<double3*>(y),
                                                                (static_cast<const double3*>(x))[0], N);
                    break;
                case 4:
                    add4_val<<<gridSize,blockSize,0,stream_id>>>(static_cast<double4*>(y),
                                                                (static_cast<const double4*>(x))[0], N);
                    break;
            }
            break;
    }
}
