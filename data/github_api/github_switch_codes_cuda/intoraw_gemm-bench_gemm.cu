// Repository: intoraw/gemm-bench
// File: gemm.cu

#include <unistd.h>
#include <iostream>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#define STRIP_FLAG_HELP 1
#include <gflags/gflags.h>

using namespace std;

DEFINE_int32(m, 1, "m");
DEFINE_int32(n, 1, "n");
DEFINE_int32(k, 1, "k");


const char* cublasGetErrorString(cublasStatus_t status)
{
    switch(status)
    {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE"; 
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH"; 
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED"; 
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR"; 
    }
    return "unknown error";
}

// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
inline
cudaError_t checkCuda(cudaError_t result)
{
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}

inline
cublasStatus_t checkCublas(cublasStatus_t result)
{
  if (result != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cublasGetErrorString(result));
    assert(result == CUBLAS_STATUS_SUCCESS);
  }
  return result;
}

void CPU_fill_rand(float*A, int nr_rows_A, int nr_cols_A){
  int a = 1;
  for (int i = 0;i < nr_rows_A * nr_cols_A; i ++){
    A[i] = (float)rand()/(float)(RAND_MAX/a);
  }
}


void bench(int m, int n, int k, int repeats){
  cublasStatus_t stat;
  cublasHandle_t handle;

  checkCublas(cublasCreate(&handle));

  float *h_A = (float*)malloc(m * k * sizeof(float));
  float *h_B = (float*)malloc(k * n * sizeof(float));
  float *h_C = (float*)malloc(m * n * sizeof(float));
  
  CPU_fill_rand(h_A, m, k);
  CPU_fill_rand(h_B, k, n);
  CPU_fill_rand(h_C, m, n);

  float *d_A, *d_B, *d_C;
  checkCuda(cudaMallocManaged(&d_A, m * k * sizeof(float)));
  checkCuda(cudaMallocManaged(&d_B, k * n * sizeof(float)));
  checkCuda(cudaMallocManaged(&d_C, m * n * sizeof(float)));

  checkCuda(cudaMemcpy(d_A, h_A, m * k * sizeof(float), cudaMemcpyHostToDevice));
  checkCuda(cudaMemcpy(d_B, h_B, k * n * sizeof(float), cudaMemcpyHostToDevice));
  checkCuda(cudaMemcpy(d_C, h_C, m * n * sizeof(float), cudaMemcpyHostToDevice));

  int lda, ldb, ldc;
  const float alf = 1.0f;
  const float bet = 0.0f;
  const float *alpha = &alf;
  const float *beta = &bet;


  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  float sum = 0.0;
  for(int rep = 0; rep < repeats; rep ++) {
    cudaEventRecord(start, 0);
    lda = m; 
    ldb = k;
    ldc = m;

    stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, d_A, lda, d_B, ldb, beta, d_C, ldc);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    if(stat != CUBLAS_STATUS_SUCCESS){
      cerr << " cublasSgemm failed" << endl;
      exit(1);
    }

    assert(!cudaGetLastError());
    float elapsed;
    cudaEventElapsedTime(&elapsed, start, stop);
    cout << m << " " << n << " " << k << " time : " << elapsed << " ms" << endl;
    sum += elapsed;
  }

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  free(h_A);
  free(h_B);
  free(h_C);
  return ;
}



int main(int argc, char **argv){
  gflags::ParseCommandLineFlags(&argc, &argv, true);


  int m, k, n;

  m = FLAGS_m;
  n = FLAGS_n;
  k = FLAGS_k;

  bench(m, n, k, 20);

  return 0;

}
