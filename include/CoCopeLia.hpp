///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief The external wrapper for CoCoPelia + wrapped cuBLASXt
///
#ifndef COCOPELIA_H
#define COCOPELIA_H

#include <cblas.h>
#include "cpu_utils.hpp"
#include "gpu_utils.hpp"

/// A pin_malloc/cudaMalloc wrapper to enable compilation without cuda for benchmarks
void * CoComalloc(long long bytes, short loc, short dev);

/// A rand/curand wrapper to enable compilation without cuda for benchmarks
void Dvec_init_CoCoRand(double *vec, long long length, int seed, short loc, short dev);
void Svec_init_CoCoRand(float *vec, long long length, int seed, short loc, short dev);

/// Transfer wrappers for CUBLAS 
void vec_get_memcpy(void* dest, void* src,  long long bytes, short dev_id);
void vec_set_memcpy(void* dest, void* src,  long long bytes, short dev_id);

/// The CoCopeLia Dgemm implementation. The input T is used if T > 0, otherwise a prediction model is used to select a tile from the micro-benchmarked tile candidates with CoCopeLia_optimize3.
void CoCopeLia_Dgemm(CBLAS_TRANSPOSE cpu_op_A,  CBLAS_TRANSPOSE cpu_op_B, size_t M, size_t N, size_t K, double alpha, double* A, size_t ldA, double* B, size_t ldB, double beta, double* C, size_t ldC, int* Tin, short dev_id);

/// The CoCopeLia Sgemm implementation. The input T is used if T > 0, otherwise a prediction model is used to select a tile from the micro-benchmarked tile candidates with CoCopeLia_optimize3.
void CoCopeLia_Sgemm(CBLAS_TRANSPOSE cpu_op_A,  CBLAS_TRANSPOSE cpu_op_B, size_t M, size_t N, size_t K, float alpha, float* A, size_t ldA, float* B, size_t ldB, float beta, float* C, size_t ldC, int* Tin, short dev_id);

/// cuBLASXt wrappers for performance evaluation
timer_p cublasXt_dgemm_wrapper(cublasOperation_t gpu_op_A,  cublasOperation_t gpu_op_B, size_t M, size_t N, size_t K, double alpha, double* A, size_t ldA, double* B, size_t ldB, double beta, double* C, size_t ldC, size_t T, double cpu_ratio, short dev_id);

timer_p cublasXt_sgemm_wrapper(cublasOperation_t gpu_op_A,  cublasOperation_t gpu_op_B, size_t M, size_t N, size_t K, float alpha, float* A, size_t ldA, float* B, size_t ldB, float beta, float* C, size_t ldC, size_t cublasXt_dim_max, double cpu_ratio, short dev_id);

/*
void CoCopeLia_sgemm(CBLAS_TRANSPOSE cpu_op_A,  CBLAS_TRANSPOSE cpu_op_B, size_t M, size_t N, size_t K, float alpha, float* A, size_t ldA, float* B, size_t ldB, float beta, float* C, size_t ldC, short dev_id);

void CoCopeLia_daxpy(size_t N, double alpha, double* x, size_t incx, double* y, size_t incy, short dev_id);

timer_p CoCopeLia_sgemm_tile(CBLAS_TRANSPOSE cpu_op_A,  CBLAS_TRANSPOSE cpu_op_B, size_t M, size_t N, size_t K, float alpha, float* A, size_t ldA, float* B, size_t ldB, float beta, float* C, size_t ldC, size_t Ms, size_t Ns, double ratio, short dev_id);

timer_p CoCopeLia_dgemv_tile(CBLAS_TRANSPOSE cpu_op_A,  size_t M, size_t N, double alpha, double* A, size_t ldA, double* x, size_t incx, double beta, double* y, size_t incy, size_t Ms, double ratio, short dev_id);

timer_p CoCopeLia_daxpy_tile(size_t N, double alpha, double* x, size_t incx, double* y, size_t incy, size_t Ns, double ratio, short dev_id);

*/
#ifdef __cplusplus
extern "C"{
#endif 
extern int xkblas_dgemm_async(
  int transA, int transB, int M, int N, int K,
  const double* alpha, const double *A, int LDA,
  const double *B, int LDB,
  const double* beta,  double *C, int LDC );

extern int xkblas_memory_coherent_async(
  int uplo, int memflag,
  size_t M, size_t N,
  void* A, size_t ld, size_t elsize
);

extern void xkblas_sync();
#ifdef __cplusplus
}
#endif


#endif
