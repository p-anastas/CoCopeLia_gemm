///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief Some CUDA function calls with added error-checking
///
#ifndef COCOBLAS_SUBKERNELS_H
#define COCOBLAS_SUBKERNELS_H

#include <cublas_v2.h>
#include <cuda_runtime.h>
 
short CoCopeLia_ptr_check_cuda_9_2(const void * in_ptr, short dev_id);

short CoCopeLia_ptr_check_cuda_11(const void * in_ptr, short dev_id);

typedef struct subkernel3_gen {
  size_t Ms, Ns, Ks;
  double* As, *Bs, *Cs;
  double* A_dev, * B_dev, *C_dev;
  short device, A_loc, B_loc, C_loc;
  size_t ldA, ldB, ldC, d_ldA, d_ldB, d_ldC; 
  short AT_master, BT_master, CT_master, CT_out_master;
  cublasOperation_t gpu_op_A, gpu_op_B; 
  cudaEvent_t data_avail, gemm_complete;
} * kernel3_p;

kernel3_p CoCopeLia_Dgemm_subkernel_init(size_t M, size_t N, size_t K, size_t Ms, size_t Ns, size_t Ks, short A_loc, short B_loc, short C_loc, short device); 
void CoCopeLia_Dgemm_subkernel_destroy(kernel3_p kernel);
//void CoCopeLia_Dgemm_subkernel_async(double alpha, double beta, kernel3_p kernel, short AT_master, short BT_master, short CT_master, short CT_out_master);
void CoCopeLia_Dgemm_subkernel_async(double alpha, double beta, kernel3_p kernel,  short d2hWaitForH2d);
void CoCopeLia_Dgemm_subkernel_out(kernel3_p kernel);

typedef struct subkernel3f_gen {
  size_t Ms, Ns, Ks;
  float* As, *Bs, *Cs;
  float* A_dev, * B_dev, *C_dev;
  short device, A_loc, B_loc, C_loc;
  size_t ldA, ldB, ldC, d_ldA, d_ldB, d_ldC; 
  short AT_master, BT_master, CT_master, CT_out_master;
  cublasOperation_t gpu_op_A, gpu_op_B; 
  cudaEvent_t data_avail, gemm_complete;
} * kernel3f_p;

kernel3f_p CoCopeLia_Sgemm_subkernel_init(size_t M, size_t N, size_t K, size_t Ms, size_t Ns, size_t Ks, short A_loc, short B_loc, short C_loc, short device); 
void CoCopeLia_Sgemm_subkernel_destroy(kernel3f_p kernel);
void CoCopeLia_Sgemm_subkernel_async(float alpha, float beta, kernel3f_p kernel, short d2hWaitForH2d);
void CoCopeLia_Sgemm_subkernel_out(kernel3f_p kernel);


/*
typedef struct subkernel3 {
  size_t Ms, Ns;
  double* As, *Bs, *Cs;
  double* A_dev, * B_dev, *C_dev;
  short device, A_loc, B_loc, C_loc;
  size_t ldA, ldB, ldC, d_ldA, d_ldB, d_ldC; 
  cublasOperation_t gpu_op_A, gpu_op_B; 
  cudaEvent_t start, setA, setB, setC, gemm, gotC, prev_send, prev_gemm;
cudaStream_t stream; 
cublasHandle_t handle;
} * kernel3_p;

typedef struct subkernelf3 {
  size_t Ms;
  size_t Ns;
  float* As, *Bs, *Cs;
  float* A_dev, * B_dev, *C_dev;
  short device, A_loc, B_loc, C_loc;
  size_t ldA, ldB, ldC, d_ldA, d_ldB, d_ldC; 
  cublasOperation_t gpu_op_A, gpu_op_B; 
  cudaEvent_t start, setA, setB, setC, gemm, gotC, prev_send, prev_gemm;
cudaStream_t stream; 
cublasHandle_t handle;
} * kernel3_f_p;

typedef struct subkernel2 {
  size_t Ms;
  double* As, *xs, *ys;
  double* A_dev, * x_dev, *y_dev;
  short device, A_loc, x_loc, y_loc;
  size_t ldA, d_ldA; 
  size_t incx,incy;
  cublasOperation_t gpu_op_A; 
  cudaEvent_t start, setA, setx, sety, gemv, goty, prev_send, prev_gemv;
cudaStream_t stream; 
cublasHandle_t handle;
} * kernel2_p;

typedef struct subkernel1 {
  size_t Ns;
  double *xs, *ys;
  double * x_dev, *y_dev;
  short device, x_loc, y_loc;
  size_t incx,incy;
  cudaEvent_t start, setx, sety, axpy, goty, prev_send, prev_axpy;
cudaStream_t stream; 
cublasHandle_t handle;
} * kernel1_p;

/// Subkernels for BLAS3 double
kernel3_p CoCoBLAS_dgemm_subkernel_init(size_t M, size_t N, size_t K, size_t Ms, size_t Ns, short A_loc, short B_loc, short C_loc, short device);

void CoCoBLAS_dgemm_subkernel_destroy(kernel3_p kernel);

void CoCoBLAS_dgemm_subkernel_async(size_t M, size_t N, size_t K, double alpha, double beta, kernel3_p kernel, short Ms_master, short Ns_master);

/// Subkernels for BLAS3 float
kernel3_f_p CoCoBLAS_sgemm_subkernel_init(size_t M, size_t N, size_t K, size_t Ms, size_t Ns, short A_loc, short B_loc, short C_loc, short device);

void CoCoBLAS_sgemm_subkernel_destroy(kernel3_f_p kernel);

void CoCoBLAS_sgemm_subkernel_async(size_t M, size_t N, size_t K, float alpha, float beta, kernel3_f_p kernel, short Ms_master, short Ns_master);

/// Subkernels for BLAS2 double
kernel2_p CoCoBLAS_dgemv_subkernel_init(size_t M, size_t N, size_t Ms, short A_loc, short x_loc, short y_loc, short device);

void CoCoBLAS_dgemv_subkernel_destroy(kernel2_p kernel);

void CoCoBLAS_dgemv_subkernel_async(size_t M, size_t N, double alpha, double beta, kernel2_p kernel, short Ms_master);

/// Subkernels for BLAS1 double

kernel1_p CoCoBLAS_daxpy_subkernel_init(size_t N, size_t Ns, short x_loc, short y_loc, short device);

void CoCoBLAS_daxpy_subkernel_destroy(kernel1_p kernel);

void CoCoBLAS_daxpy_subkernel_async(size_t N, double alpha, kernel1_p kernel);


typedef struct subkernel3 {
  size_t T;
  double* As, *Bs, *Cs;
  double* A_dev, * B_dev, *C_dev;
  short device, A_loc, B_loc, C_loc;
  size_t ldA, ldB, ldC, d_ldA, d_ldB, d_ldC; 
  cublasOperation_t gpu_op_A, gpu_op_B; 
  cudaEvent_t start, setA, setB, setC, gemm, gotC, prev_send, prev_gemm;
cudaStream_t stream; 
cublasHandle_t handle;
} * kernel3T_p;

kernel3T_p CoCopeLia_dgemm_Tiled_subkernel_init(size_t M, size_t N, size_t K, size_t T, short A_loc, short B_loc, short C_loc, short device);
void CoCopeLia_dgemm_Tiled_subkernel_destroy(kernel3T_p kernel);
void CoCopeLia_dgemm_Tiled_subkernel_async(double alpha, double beta, kernel3T_p kernel, short AT_master, short BT_master, short CT_master, short CT_out_master);

*/

#endif
