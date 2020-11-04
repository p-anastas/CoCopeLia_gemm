///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief The start of Zawarudo
///

#include "cpu_utils.hpp"
#include "CoCopeLia.hpp"
#include "testing.hpp"

int main(const int argc, const char *argv[]) {

  	float alpha, beta;
  	alpha = 1.1234, beta = 1.2345;

  	int ctr = 1;

	char machine[256];
	size_t M = 0, N = 0, K = 0;

	short A_loc = 1, B_loc = 1, C_loc = 1, dev_id = -1; 
	switch (argc) {
	case (9):
		A_loc = atoi(argv[argc-3]);
		B_loc = atoi(argv[argc-2]);
		C_loc = atoi(argv[argc-1]);
	case (6):
		sprintf(machine , "%s", argv[ctr++]);
		M = atoi(argv[ctr++]);
		N = atoi(argv[ctr++]);
		K = atoi(argv[ctr++]);
		dev_id = atoi(argv[ctr++]);
		break;
	default:
		error("Incorrect input arguments. Usage: ./correct_run machine M N K dev_id [A_loc B_loc C_loc]");
  	}

	if (strcmp(MACHINE, machine)) error("Serial_Sgemm: Running on wrong machine");
	else fprintf(stderr, "Problem configuration (M=%zu, N=%zu, K=%zu) on locations (A_loc=%d, B_loc=%d, C_loc=%d)\n", M, N, K, A_loc, B_loc, C_loc);

	char *filename = (char *) malloc(256* sizeof(char));
	sprintf(filename, "%s/Data_manipulation/Results/%s/evaluation/Serial_Sgemm_%d.log", PROJECTDIR, MACHINE, dev_id);
	check_log_ser(filename, M, N, K, A_loc, B_loc, C_loc, dev_id);

	size_t ldA = M, ldB = K, ldC = M;

	/// Matrix Layouts for GPU GEMM
	cublasOperation_t gpu_op_A = CUBLAS_OP_N, gpu_op_B = CUBLAS_OP_N;  // CUBLAS_OP_N, CUBLAS_OP_T

	/// Local Timers 
	double cpu_timer = csecond();

	fprintf(stderr, "\nAllocating memory...");

	float *A, *B, *C;
	// allocate in device if loc = 0, otherwise allocate in pinned memory for benchmarks
	A = (float*) CoComalloc(M * K*sizeof(float), A_loc, dev_id);
	B = (float*) CoComalloc(N * K*sizeof(float), B_loc, dev_id);
	C = (float*) CoComalloc(M * N*sizeof(float), C_loc, dev_id);

	cudaCheckErrors();
	cpu_timer  = csecond() - cpu_timer;
	fprintf(stderr, "done.\nAlloc time:\t%lf ms\n\n",  cpu_timer  * 1000);

#ifdef VALIDATE
	cpu_timer = csecond();
	fprintf(stderr, "Initializing to random values (VALIDATE)..."); 
	Svec_init_CoCoRand(A, K * M, 42, A_loc, dev_id);
	Svec_init_CoCoRand(B, K * N, 43, B_loc, dev_id);
	Svec_init_CoCoRand(C, M * N, 44, C_loc, dev_id);
	cudaCheckErrors();
	cpu_timer  = csecond() - cpu_timer ;
	fprintf(stderr, "done.\nInit time:\t%lf ms\n\n",  cpu_timer  * 1000);

	float *C_out, *C_out1;
	C_out  = (float*) malloc(M * N*sizeof(float));
	C_out1  = (float*) malloc(M * N*sizeof(float));

 	if (C_loc) optimized_memcpy(C_out1, C,  M * N *sizeof(float));
 	else vec_get_memcpy(C_out1, C,  M * N *sizeof(float), dev_id);
#endif

	cpu_timer = csecond();
	fprintf(stderr, "Extra GPU allocation for experiment..."); 
	float *A_dev, *B_dev, *C_dev; 
 	if (A_loc) A_dev = (float*) CoComalloc(M * K*sizeof(float), 0, dev_id);
	else A_dev = A; 
 	if (B_loc) B_dev = (float*) CoComalloc(N * K*sizeof(float), 0, dev_id);
	else B_dev = B; 
 	if (C_loc) C_dev = (float*) CoComalloc(M * N*sizeof(float), 0, dev_id);
	else C_dev = C; 
	cublasHandle_t handle;
	massert(CUBLAS_STATUS_SUCCESS == cublasCreate(&handle), "Serial_Sgemm: NOT CUBLAS_STATUS_SUCCESS");
	cudaCheckErrors();
	cpu_timer  = csecond() - cpu_timer ;
	fprintf(stderr, "done.\nExtra alloc time:\t%lf ms\n\n",  cpu_timer  * 1000);


	fprintf(stderr, "Sending Data to device..."); 
	double send_t = csecond();
 	if (A_loc) vec_set_memcpy(A_dev, A,  M * K *sizeof(float), dev_id);
 	if (B_loc) vec_set_memcpy(B_dev, B,  N * K *sizeof(float), dev_id);
 	if (C_loc) vec_set_memcpy(C_dev, C,  M * N *sizeof(float), dev_id);
	cudaCheckErrors();
	send_t = csecond() - send_t; 
	fprintf(stderr, "done.\nH2D time:\t%lf ms\n\n",  send_t  * 1000);

	// First call for Validate and/or additional overhead counting
	cpu_timer = csecond();
	massert(CUBLAS_STATUS_SUCCESS == cublasSgemm(handle, gpu_op_A, gpu_op_B, M, N, K, &alpha, A_dev, ldA, B_dev, ldB, &beta, C_dev, ldC), "Serial_Sgemm: NOT CUBLAS_STATUS_SUCCESS");
	cudaCheckErrors();
	cpu_timer  = csecond() - cpu_timer;
	fprintf(stderr,"First Serial SGEMM call-> M = %zu, N = %zu, K = %zu\n", M, N, K);
	fprintf(stderr, "GPU exec time:\t%lf ms\n\n", cpu_timer  * 1000);
	double gpu_first_t = cpu_timer; 

	fprintf(stderr, "Getting Result from device(if needed)..."); 
	double get_t = csecond();
 	if (C_loc) vec_get_memcpy(C, C_dev,  M * N *sizeof(float), dev_id);
	cudaCheckErrors();
	get_t = csecond() - get_t; 
	fprintf(stderr, "done.\nD2H time:\t%lf ms\n\n",  get_t  * 1000);


#ifdef VALIDATE
 	if (C_loc) optimized_memcpy(C_out, C,  M * N *sizeof(float));
 	else vec_get_memcpy(C_out, C,  M * N *sizeof(float), dev_id);
#endif

	fprintf(stderr,"Running Serial SGEMM-> M = %zu, N = %zu, K = %zu\n", M, N, K);
	double min_t = gpu_first_t, max_t = 0, avg_t = 0;
	cpu_timer = csecond();
	short bench_it = 100;
	if ( M >= 8192 || N >= 8192 || K >= 8192) bench_it = 10; 
	for(int it = 0; it < bench_it; it++){
		cpu_timer = csecond();
		massert(CUBLAS_STATUS_SUCCESS == cublasSgemm(handle, gpu_op_A, gpu_op_B, M, N, K, &alpha, A_dev, ldA, B_dev, ldB, &beta, C_dev, ldC), "Serial_Sgemm: NOT CUBLAS_STATUS_SUCCESS");
		cudaCheckErrors();
		cpu_timer = csecond() - cpu_timer;
		if ( cpu_timer < min_t ) min_t = cpu_timer;
		if ( cpu_timer > max_t ) max_t = cpu_timer;
		avg_t += cpu_timer;
	}
	avg_t/=bench_it;
	fprintf(stderr, "\nCoCopeLia :\n\tavg_t = %lf ms ( %lf Gflops/s )\n\tmin_t = %lf ms ( %lf Gflops/s )\n\tmax_t = %lf ms ( %lf Gflops/s )\n", 
	(avg_t + send_t + get_t) * 1000, Gval_per_s(dgemm_flops(M,N,K),avg_t + send_t + get_t),
	(min_t + send_t + get_t) * 1000, Gval_per_s(dgemm_flops(M,N,K),min_t + send_t + get_t),
	(max_t + send_t + get_t) * 1000, Gval_per_s(dgemm_flops(M,N,K),max_t + send_t + get_t));

#ifdef VALIDATE
 	if (C_loc) optimized_memcpy(C, C_out1,  M * N *sizeof(float));
 	else vec_set_memcpy(C, C_out1,  M * N *sizeof(float), dev_id);
	cublasXt_sgemm_wrapper(gpu_op_A,  gpu_op_B, M, N, K, alpha, A, ldA, B, ldB, beta, C, ldC,  (size_t) fmin(fmin(M,N),K)/2, 0, dev_id);
 	if (C_loc) optimized_memcpy(C_out1, C,  M * N *sizeof(float));
	else vec_get_memcpy(C_out1, C,  M * N *sizeof(float), dev_id);
 	//if(Stest_equality(C_out1, C_out, M * N) < 1) error("Insufficient accuracy for benchmarks");
#endif

	report_results_ser(filename, M, N, K, A_loc, B_loc, C_loc, dev_id, avg_t, min_t, max_t, gpu_first_t, send_t, get_t); 
	cudaCheckErrors();

	return 0;
}
