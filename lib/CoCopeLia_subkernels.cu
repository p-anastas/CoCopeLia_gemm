///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief The CoCopeLia sub-kernel Tile scheduler functions. 
///

#include <cassert>

#include "CoCopeLia_subkernels.hpp"
#include "cpu_utils.hpp"
#include "gpu_utils.hpp"

cudaStream_t h2d_stream = NULL, d2h_stream = NULL, exec_stream = NULL;
cublasHandle_t handle;
cudaEvent_t h2d_complete = NULL; 

/// Checks if given ptr in CPU (pinned) or GPU. FIXME: See next function for CUDA > 10.2
short CoCopeLia_ptr_check_cuda_9_2(const void * in_ptr, short dev_id)
{
	short loc = -1;
	cudaPointerAttributes ptr_att; 
	if (cudaSuccess != cudaPointerGetAttributes(&ptr_att, in_ptr)){
		warning("CoCoBLAS_ptr_check_cuda_9_2: Pointer not visible to CUDA, host alloc or error");
		cudaCheckErrors();
	}
	if (ptr_att.device != dev_id) error("CoCoBLAS_ptr_check_cuda_9_2: Pointer and target device don't match");
	if (ptr_att.memoryType == cudaMemoryTypeHost) loc = 1; 
	else if (ptr_att.memoryType == cudaMemoryTypeDevice) loc = 0;
	// TODO: Unified memory is considered available in the GPU as cuBLASXt ( not bad, not great) 
	else if (ptr_att.isManaged) loc = 0;
	else error("CoCoBLAS_ptr_check_cuda_9_2: Invalid memory type");
	return loc; 
}


short CoCopeLia_ptr_check_cuda_11(const void * in_ptr, short dev_id)
{
	short loc = -1;
	cudaPointerAttributes ptr_att; 
	if (cudaSuccess != cudaPointerGetAttributes(&ptr_att, in_ptr)){
		warning("CoCoBLAS_ptr_check_cuda_9_2: Pointer not visible to CUDA, host alloc or error");
		cudaCheckErrors();
	}
	error("CoCoBLAS_ptr_check_cuda_11: Uncomment part in /lib/CoCopeLia_subkernels.cu to include unified memory flag for latest cuda");
}
/* Did you say "Backward compatibility"? Probably not! Use this for CUDA > 10.2
	if (ptr_att.device != dev_id) error("CoCoBLAS_ptr_check_cuda_11: Pointer and target device don't match");
	if (ptr_att.type == cudaMemoryTypeHost) loc = 1;
	else if (ptr_att.type == cudaMemoryTypeDevice) loc = 0;
	// TODO: Unified memory is considered available in the GPU as cuBLASXt ( not bad, not great) 
	else if (ptr_att.type == cudaMemoryTypeManaged) loc = 0;
	else error("CoCoBLAS_ptr_check_cuda_11: Invalid memory type");
	return loc; 
}
*/

/// Initializes a Dgemm subkernel with given dimensions, and creates the 3 overlap CUDA streams if needed. 
kernel3_p CoCopeLia_Dgemm_subkernel_init(size_t M, size_t N, size_t K, size_t Ms, size_t Ns, size_t Ks, short A_loc, short B_loc, short C_loc, short device) {
  	
	kernel3_p kernel = (kernel3_p)malloc(sizeof(struct subkernel3_gen));

	kernel->device = device;
	kernel->A_loc = A_loc;
	kernel->B_loc = B_loc;
	kernel->C_loc = C_loc;

  	kernel->Ms = Ms;
  	kernel->Ns = Ns;
  	kernel->Ks = Ks;

	kernel->ldA = M;
	kernel->d_ldA = M;
	kernel->gpu_op_A = CUBLAS_OP_N;

	kernel->ldB = K;
	kernel->d_ldB = K;
	kernel->gpu_op_B = CUBLAS_OP_N;

	kernel->ldC = M;
	kernel->d_ldC = M;


  	if (!h2d_stream) cudaStreamCreate(&h2d_stream);
  	if (!d2h_stream) cudaStreamCreate(&d2h_stream);
  	if (!exec_stream) cudaStreamCreate(&exec_stream);
	if (!handle){
		assert(CUBLAS_STATUS_SUCCESS == cublasCreate(&handle));
		assert(CUBLAS_STATUS_SUCCESS == cublasSetStream(handle, exec_stream));
	}

	if (!h2d_complete) cudaEventCreateWithFlags(&h2d_complete, cudaEventDisableTiming);
	cudaEventCreateWithFlags(&kernel->data_avail, cudaEventDisableTiming);
	cudaEventCreateWithFlags(&kernel->gemm_complete, cudaEventDisableTiming);

return kernel;
}

/// Destroys given Dgemm subkernel.
void CoCopeLia_Dgemm_subkernel_destroy(kernel3_p kernel){
	// TODO: For now use only one device;
	int dev_id; cudaGetDevice(&dev_id);

	assert(CUBLAS_STATUS_SUCCESS == cudaEventDestroy(kernel->data_avail));
	assert(CUBLAS_STATUS_SUCCESS == cudaEventDestroy(kernel->gemm_complete));
	
}

/// Puts a Dgemm sub-kernel's input and execution on the corresponding stream pipelines.
void CoCopeLia_Dgemm_subkernel_async(double alpha, double beta, kernel3_p kernel, short d2hWaitForH2d){
	if(!kernel->A_dev && alpha)error("CoCoBLAS_Dgemm_subkernel_async: A_dev buffer unallocated");
	else if(!kernel->B_dev && alpha )error("CoCoBLAS_Dgemm_subkernel_async: B_dev buffer unallocated");
	else if(!kernel->C_dev)error("CoCoBLAS_Dgemm_subkernel_async: C_dev buffer unallocated");

	if (kernel->AT_master && kernel->A_loc && alpha) assert(CUBLAS_STATUS_SUCCESS == cublasSetMatrixAsync(kernel->Ms, kernel->Ks, sizeof(double), kernel->As, kernel->ldA, kernel->A_dev, kernel->d_ldA, h2d_stream));
	if (kernel->BT_master && kernel->B_loc && alpha) assert(CUBLAS_STATUS_SUCCESS == cublasSetMatrixAsync(kernel->Ks, kernel->Ns, sizeof(double), kernel->Bs, kernel->ldB, kernel->B_dev, kernel->d_ldB, h2d_stream));
	if (kernel->CT_master && kernel->C_loc && beta) assert(CUBLAS_STATUS_SUCCESS == cublasSetMatrixAsync(kernel->Ms, kernel->Ns, sizeof(double), kernel->Cs, kernel->ldC, kernel->C_dev, kernel->d_ldC, h2d_stream));
	cudaEventRecord(kernel->data_avail, h2d_stream);
	//cudaCheckErrors();
	
	if (d2hWaitForH2d) cudaEventRecord(h2d_complete, h2d_stream);

	cudaStreamWaitEvent(exec_stream, kernel->data_avail,0);
	assert(CUBLAS_STATUS_SUCCESS == cublasDgemm(handle, kernel->gpu_op_A, kernel->gpu_op_B, kernel->Ms, kernel->Ns, kernel->Ks, &alpha, kernel->A_dev, kernel->d_ldA, kernel->B_dev, kernel->d_ldB, &beta, kernel->C_dev, kernel->d_ldC));
	if (kernel->CT_out_master) cudaEventRecord(kernel->gemm_complete, exec_stream);
	//cudaCheckErrors();

	return ;
}

/// Puts a Dgemm sub-kernel's output on the corresponding stream pipeline.
void CoCopeLia_Dgemm_subkernel_out(kernel3_p kernel)
{
	cudaStreamWaitEvent(d2h_stream, kernel->gemm_complete,0);
	cudaStreamWaitEvent(d2h_stream, h2d_complete,0);
	assert(CUBLAS_STATUS_SUCCESS == cublasGetMatrixAsync(kernel->Ms, kernel->Ns, sizeof(double), kernel->C_dev, kernel->d_ldC, kernel->Cs, kernel->ldC, d2h_stream));
	//cudaCheckErrors();
}

/// Initializes an Sgemm subkernel with given dimensions, and creates the 3 overlap CUDA streams if needed. 
kernel3f_p CoCopeLia_Sgemm_subkernel_init(size_t M, size_t N, size_t K, size_t Ms, size_t Ns, size_t Ks, short A_loc, short B_loc, short C_loc, short device) {
  	
	kernel3f_p kernel = (kernel3f_p)malloc(sizeof(struct subkernel3f_gen));

	kernel->device = device;
	kernel->A_loc = A_loc;
	kernel->B_loc = B_loc;
	kernel->C_loc = C_loc;

  	kernel->Ms = Ms;
  	kernel->Ns = Ns;
  	kernel->Ks = Ks;

	kernel->ldA = M;
	kernel->d_ldA = M;
	kernel->gpu_op_A = CUBLAS_OP_N;

	kernel->ldB = K;
	kernel->d_ldB = K;
	kernel->gpu_op_B = CUBLAS_OP_N;

	kernel->ldC = M;
	kernel->d_ldC = M;


  	if (!h2d_stream) cudaStreamCreate(&h2d_stream);
  	if (!d2h_stream) cudaStreamCreate(&d2h_stream);
  	if (!exec_stream) cudaStreamCreate(&exec_stream);
	if (!handle){
		assert(CUBLAS_STATUS_SUCCESS == cublasCreate(&handle));
		assert(CUBLAS_STATUS_SUCCESS == cublasSetStream(handle, exec_stream));
	}

	if (!h2d_complete) cudaEventCreateWithFlags(&h2d_complete, cudaEventDisableTiming);
	cudaEventCreateWithFlags(&kernel->data_avail, cudaEventDisableTiming);
	cudaEventCreateWithFlags(&kernel->gemm_complete, cudaEventDisableTiming);

return kernel;
}

/// Destroys given Sgemm subkernel.
void CoCopeLia_Sgemm_subkernel_destroy(kernel3f_p kernel){
	// TODO: For now use only one device;
	int dev_id; cudaGetDevice(&dev_id);

	assert(CUBLAS_STATUS_SUCCESS == cudaEventDestroy(kernel->data_avail));
	assert(CUBLAS_STATUS_SUCCESS == cudaEventDestroy(kernel->gemm_complete));
	
}

/// Puts an Sgemm sub-kernel's input and execution on the corresponding stream pipelines.
void CoCopeLia_Sgemm_subkernel_async(float alpha, float beta, kernel3f_p kernel, short d2hWaitForH2d){
	if(!kernel->A_dev && alpha)error("CoCoBLAS_Dgemm_subkernel_async: A_dev buffer unallocated");
	else if(!kernel->B_dev && alpha )error("CoCoBLAS_Dgemm_subkernel_async: B_dev buffer unallocated");
	else if(!kernel->C_dev)error("CoCoBLAS_Dgemm_subkernel_async: C_dev buffer unallocated");

	if (kernel->AT_master && kernel->A_loc && alpha) assert(CUBLAS_STATUS_SUCCESS == cublasSetMatrixAsync(kernel->Ms, kernel->Ks, sizeof(float), kernel->As, kernel->ldA, kernel->A_dev, kernel->d_ldA, h2d_stream));
	if (kernel->BT_master && kernel->B_loc && alpha) assert(CUBLAS_STATUS_SUCCESS == cublasSetMatrixAsync(kernel->Ks, kernel->Ns, sizeof(float), kernel->Bs, kernel->ldB, kernel->B_dev, kernel->d_ldB, h2d_stream));
	if (kernel->CT_master && kernel->C_loc && beta) assert(CUBLAS_STATUS_SUCCESS == cublasSetMatrixAsync(kernel->Ms, kernel->Ns, sizeof(float), kernel->Cs, kernel->ldC, kernel->C_dev, kernel->d_ldC, h2d_stream));
	cudaEventRecord(kernel->data_avail, h2d_stream);
	//cudaCheckErrors();
	
	if (d2hWaitForH2d) cudaEventRecord(h2d_complete, h2d_stream);

	cudaStreamWaitEvent(exec_stream, kernel->data_avail,0);
	assert(CUBLAS_STATUS_SUCCESS == cublasSgemm(handle, kernel->gpu_op_A, kernel->gpu_op_B, kernel->Ms, kernel->Ns, kernel->Ks, &alpha, kernel->A_dev, kernel->d_ldA, kernel->B_dev, kernel->d_ldB, &beta, kernel->C_dev, kernel->d_ldC));
	if (kernel->CT_out_master) cudaEventRecord(kernel->gemm_complete, exec_stream);
	//cudaCheckErrors();

	return ;
}

/// Puts an Sgemm sub-kernel's output on the corresponding stream pipeline.
void CoCopeLia_Sgemm_subkernel_out(kernel3f_p kernel)
{
	cudaStreamWaitEvent(d2h_stream, kernel->gemm_complete,0);
	cudaStreamWaitEvent(d2h_stream, h2d_complete,0);
	assert(CUBLAS_STATUS_SUCCESS == cublasGetMatrixAsync(kernel->Ms, kernel->Ns, sizeof(float), kernel->C_dev, kernel->d_ldC, kernel->Cs, kernel->ldC, d2h_stream));
	//cudaCheckErrors();
}
