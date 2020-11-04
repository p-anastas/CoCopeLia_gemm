///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief BLAS lvl 3 wrappers for benchmarks. 
///
#include <cassert>
#include <cublasXt.h>
#include <cblas.h>

#include "cpu_utils.hpp"
#include "gpu_utils.hpp"

void cblas_dgemm_wrap_for_cublasXt(char* gpu_op_A, char* gpu_op_B, int* M, int* N, int* K, double* alpha, double* A, int* ldA, double* B, int* ldB, double* beta, double* C, int* ldC){ 
  CBLAS_TRANSPOSE cpu_op_A, cpu_op_B;    // CblasNoTrans, CblasTrans

    //fprintf(stderr, "%d %d %d %lf %d %d %lf %d\n",*M, *N, *K, *alpha, *ldA, *ldB, *beta, *ldC);

    if(*gpu_op_A == 'N') cpu_op_A = CblasNoTrans;
    else if(*gpu_op_A == 'T') cpu_op_A = CblasTrans;
    else error("cblas_dgemm_wrap -> Invalid CUBLAS_OP for A");
    if(*gpu_op_B == 'N') cpu_op_B = CblasNoTrans;
    else if(*gpu_op_B == 'T') cpu_op_B = CblasTrans;
    else error("cblas_dgemm_wrap -> Invalid CUBLAS_OP for B");

cblas_dgemm(CblasColMajor, cpu_op_A, cpu_op_B, *M, *N, *K, *alpha, A, *ldA, B, *ldB, *beta, C, *ldC);
}

timer_p cublasXt_dgemm_wrapper(cublasOperation_t gpu_op_A,  cublasOperation_t gpu_op_B, size_t M, size_t N, size_t K, double alpha, double* A, size_t ldA, double* B, size_t ldB, double beta, double* C, size_t ldC, size_t cublasXt_dim_max, double cpu_ratio, short dev_id){
  	timer_p timer = init_timer();

	cublasStatus_t stat;
	cublasXtHandle_t handle0;
	int device_ids[1] = {dev_id};

	// TODO: For now use only one device;
	int cur_id; cudaGetDevice(&cur_id);
	if ( cur_id != dev_id) printf("cublasXt_dgemm_wrapper: Device change initiated(%d->%d)\n",cur_id, dev_id);
	cudaSetDevice(dev_id);

	/// Required allocations for device
	timer->alloc_t = csecond();
	assert(CUBLAS_STATUS_SUCCESS == cublasXtCreate(&handle0));
	assert(CUBLAS_STATUS_SUCCESS == cublasXtDeviceSelect(handle0, 1, device_ids));
	assert(CUBLAS_STATUS_SUCCESS == cublasXtSetBlockDim(handle0, cublasXt_dim_max));
	assert(CUBLAS_STATUS_SUCCESS == cublasXtSetCpuRoutine(handle0, CUBLASXT_GEMM, CUBLASXT_DOUBLE, (void*) &cblas_dgemm_wrap_for_cublasXt));
	assert(CUBLAS_STATUS_SUCCESS == cublasXtSetCpuRatio(handle0, CUBLASXT_GEMM, CUBLASXT_DOUBLE, cpu_ratio));
	assert(CUBLAS_STATUS_SUCCESS == cublasXtSetPinningMemMode(handle0, CUBLASXT_PINNING_ENABLED));
	timer->alloc_t = csecond() - timer->alloc_t;

    	timer->total_t = csecond();
	assert(CUBLAS_STATUS_SUCCESS == cublasXtDgemm(handle0, gpu_op_A, gpu_op_B, M, N, K, &alpha, A, ldA, B, ldB, &beta, C, ldC));
	cudaCheckErrors();
	timer->total_t = csecond() - timer->total_t;

	/// Free local buffers
	cublasXtDestroy(handle0);
	cudaCheckErrors();
	return timer;

}

void cblas_sgemm_wrap_for_cublasXt(char* gpu_op_A, char* gpu_op_B, int* M, int* N, int* K, float* alpha, float* A, int* ldA, float* B, int* ldB, float* beta, float* C, int* ldC){ 
  CBLAS_TRANSPOSE cpu_op_A, cpu_op_B;    // CblasNoTrans, CblasTrans

    //fprintf(stderr, "%d %d %d %lf %d %d %lf %d\n",*M, *N, *K, *alpha, *ldA, *ldB, *beta, *ldC);

    if(*gpu_op_A == 'N') cpu_op_A = CblasNoTrans;
    else if(*gpu_op_A == 'T') cpu_op_A = CblasTrans;
    else error("cblas_dgemm_wrap -> Invalid CUBLAS_OP for A");
    if(*gpu_op_B == 'N') cpu_op_B = CblasNoTrans;
    else if(*gpu_op_B == 'T') cpu_op_B = CblasTrans;
    else error("cblas_dgemm_wrap -> Invalid CUBLAS_OP for B");

cblas_sgemm(CblasColMajor, cpu_op_A, cpu_op_B, *M, *N, *K, *alpha, A, *ldA, B, *ldB, *beta, C, *ldC);
}

timer_p cublasXt_sgemm_wrapper(cublasOperation_t gpu_op_A,  cublasOperation_t gpu_op_B, size_t M, size_t N, size_t K, float alpha, float* A, size_t ldA, float* B, size_t ldB, float beta, float* C, size_t ldC, size_t cublasXt_dim_max, double cpu_ratio, short dev_id){
  	timer_p timer = init_timer();

	cublasStatus_t stat;
	cublasXtHandle_t handle0;
	int device_ids[1] = {dev_id};

	// TODO: For now use only one device;
	int cur_id; cudaGetDevice(&cur_id);
	if ( cur_id != dev_id) printf("cublasXt_dgemm_wrapper: Device change initiated(%d->%d)\n",cur_id, dev_id);
	cudaSetDevice(dev_id);

	/// Required allocations for device
	timer->alloc_t = csecond();
	assert(CUBLAS_STATUS_SUCCESS == cublasXtCreate(&handle0));
	assert(CUBLAS_STATUS_SUCCESS == cublasXtDeviceSelect(handle0, 1, device_ids));
	assert(CUBLAS_STATUS_SUCCESS == cublasXtSetBlockDim(handle0, cublasXt_dim_max));
	assert(CUBLAS_STATUS_SUCCESS == cublasXtSetCpuRoutine(handle0, CUBLASXT_GEMM, CUBLASXT_FLOAT, (void*) &cblas_sgemm_wrap_for_cublasXt));
	assert(CUBLAS_STATUS_SUCCESS == cublasXtSetCpuRatio(handle0, CUBLASXT_GEMM, CUBLASXT_FLOAT, cpu_ratio));
	assert(CUBLAS_STATUS_SUCCESS == cublasXtSetPinningMemMode(handle0, CUBLASXT_PINNING_ENABLED));
	timer->alloc_t = csecond() - timer->alloc_t;

    	timer->total_t = csecond();
	assert(CUBLAS_STATUS_SUCCESS == cublasXtSgemm(handle0, gpu_op_A, gpu_op_B, M, N, K, &alpha, A, ldA, B, ldB, &beta, C, ldC));
	cudaCheckErrors();
	timer->total_t = csecond() - timer->total_t;

	/// Free local buffers
	cublasXtDestroy(handle0);
	cudaCheckErrors();
	return timer;

}
