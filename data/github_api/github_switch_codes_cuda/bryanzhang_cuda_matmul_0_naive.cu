// Repository: bryanzhang/cuda_matmul
// File: 0_naive.cu

#include <cstdio>
#include <cstdlib>
#include <cmath>

#include <iostream>

#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <cublas_v2.h>

#define CUDA_CALL(func) \
    do { \
      cudaError_t err = (func); \
      if(err != cudaSuccess) { \
        printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
      } \
    } while(0)


// 每个thread block内线程为BLOCKSIZE * BLOCKSIZE
#define BLOCK_SIZE_M 32
#define BLOCK_SIZE_N 32
#define BLOCK_SIZE_K 32

#define VLX 32
#define VLY 32

template <int M, int N, int K>
__global__ void MatMulKernel(float* A, float* B, float* C) {
  float value = 0.0;
  int row = blockIdx.y * BLOCK_SIZE_M + threadIdx.y;
  int col = blockIdx.x * BLOCK_SIZE_N + threadIdx.x;
#pragma unroll
  for (int i = 0; i < K; ++i) {
    value += A[row * K + i] * B[i * N + col];
  }
  C[row * N + col] = value;
}

enum Strategy {
  CUBLAS,
  VANILLA,
  SHARED_MEMORY,
  REDUCE_BANK_CONFLICTS,
};

template <int M, int K, int N, bool enableProfiler>
float testMatMul(Strategy st, bool checkResult) {
  float* h_A, *h_B, *h_C;
  float* d_A, *d_B, *d_C;

  // 分配主机内存.
  h_A = (float*)malloc(M * K * sizeof(float));
  h_B = (float*)malloc(K * N * sizeof(float));
  h_C = (float*)malloc(M * N * sizeof(float));

  // 分配设备内存
  CUDA_CALL(cudaMalloc((void**)&d_A, M * K * sizeof(float)));
  CUDA_CALL(cudaMalloc((void**)&d_B, K * N * sizeof(float)));
  CUDA_CALL(cudaMalloc((void**)&d_C, M * N * sizeof(float)));

  // 初始化主机内存
  for (int i = 0; i < M * K; ++i) {
    h_A[i] = rand() / (float)RAND_MAX;
  }
  for (int i = 0; i < K * N; ++i) {
    h_B[i] = rand() / (float)RAND_MAX;
  }

  // 将数据从主机内存拷贝到设备内存
  CUDA_CALL(cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice));

  // 创建CUDA事件
  cudaEvent_t start, stop;
  CUDA_CALL(cudaEventCreate(&start));
  CUDA_CALL(cudaEventCreate(&stop));

  dim3 dimBlock(BLOCK_SIZE_N, BLOCK_SIZE_M);
  dim3 dimGrid(N / BLOCK_SIZE_N, M / BLOCK_SIZE_M);
  CUDA_CALL(cudaEventRecord(start));
  if (enableProfiler) {
    cudaProfilerStart();
  }
  MatMulKernel<M, N, K><<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
      printf("Error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }

  CUDA_CALL(cudaDeviceSynchronize());
  if (enableProfiler) {
    cudaProfilerStop();
  }
  CUDA_CALL(cudaEventRecord(stop));
  CUDA_CALL(cudaEventSynchronize(stop));

  float milliseconds = 0;
  CUDA_CALL(cudaEventElapsedTime(&milliseconds, start, stop));

  // 清理
  free(h_A);
  free(h_B);
  free(h_C);
  CUDA_CALL(cudaFree(d_A));
  CUDA_CALL(cudaFree(d_B));
  CUDA_CALL(cudaFree(d_C));
  CUDA_CALL(cudaEventDestroy(start));
  CUDA_CALL(cudaEventDestroy(stop));
  return milliseconds;
}

const char* strategyToString(Strategy st) {
  switch (st) {
    case CUBLAS:
      return "CUBLAS";
    case VANILLA:
      return "VANILLA";
    case SHARED_MEMORY:
      return "SHARED_MEMORY";
    case REDUCE_BANK_CONFLICTS:
      return "REDUCE_BANK_CONFLICTS";
    default:
      return "Unknown";
  }
}

int main() {
  cudaDeviceProp prop;
  int count;
  cudaGetDeviceCount(&count);
  for (int i = 0; i < count; i++) {
    cudaGetDeviceProperties(&prop, i);
    std::cout << "Device " << i << ":\n";
    std::cout << "Maximum threads per block: " << prop.maxThreadsPerBlock << "\n";
    std::cout << "Maximum dimension size of a thread block: (" << prop.maxThreadsDim[0] << ", " << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2] << ")\n";
    std::cout << "Maximum dimension size of a grid size: (" << prop.maxGridSize[0] << ", " << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << ")\n";
  }

  constexpr int times = 30;
  constexpr Strategy st = REDUCE_BANK_CONFLICTS;
  constexpr int M = 3840, K = 2880, N = 3840;
  constexpr bool checkResult = false, enableProfiler = false;
  float accMillis = 0.0;
  for (int i = 0; i <  times; ++i) {
    accMillis += testMatMul<M, K, N, enableProfiler>(st, checkResult);
    if (((i + 1) % 10) == 0) {
      printf("Testing process: %d / %d\n", (i + 1), times);
    }
  }
  printf("M=%d, K=%d, N=%d, strategy=%s, bs_m=%d, bs_n=%d, bs_k=%d, enableProfiler=%d, MatMul: Totally elapsed time in GPU was %.2f ms, %.2f ms per operation\n",
                  M, K, N, strategyToString(st), BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, enableProfiler ? 1: 0, accMillis, accMillis / times);
}
