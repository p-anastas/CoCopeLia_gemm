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
	size_t M = -1, N = -1, K = -1;

	int Tin = -1; 

	short A_loc = 1, B_loc = 1, C_loc = 1, dev_id = -1; 
	switch (argc) {
	case (10):
		A_loc = atoi(argv[argc-3]);
		B_loc = atoi(argv[argc-2]);
		C_loc = atoi(argv[argc-1]);
	case (7):
		sprintf(machine , "%s", argv[ctr++]);
		M = atoi(argv[ctr++]);
		N = atoi(argv[ctr++]);
		K = atoi(argv[ctr++]);
		Tin = atoi(argv[ctr++]);
		dev_id = atoi(argv[ctr++]);
		break;
	default:
		error("Incorrect input arguments. Usage: ./correct_run machine M N K dev_id [A_loc B_loc C_loc]");
  	}

	if (strcmp(MACHINE, machine)) error("cuBLASXt_Sgemm: Running on wrong machine");
	else fprintf(stderr, "Problem configuration (M=%zu, N=%zu, K=%zu) on locations (A_loc=%d, B_loc=%d, C_loc=%d)\n", M, N, K, A_loc, B_loc, C_loc);

	char *filename = (char *) malloc(256* sizeof(char));
	if (Tin > 0){
		if (Tin > M || Tin > N || Tin > K) error("Given Tin bigger than problem dim"); 
		else if (Tin > M/1.5 && Tin > N/1.5 && Tin > K/1.5) error("Given Tin bigger than all problem dims/1.5");
		sprintf(filename, "%s/Data_manipulation/Results/%s/validation/cuBLASXt_Sgemm_%d.log", PROJECTDIR, MACHINE, dev_id);	
		check_log(filename, M, N, K, Tin, A_loc, B_loc, C_loc, dev_id);
	}
	else {
		sprintf(filename, "%s/Data_manipulation/Results/%s/evaluation/cuBLASXt_Sgemm_%d.log", PROJECTDIR, MACHINE, dev_id);
		check_log(filename, M, N, K, fmin(fmin(M,N),K)/2, A_loc, B_loc, C_loc, dev_id);
		for (size_t T_trial = 1024; T_trial <= (size_t) fmin(fmin(fmin(M,N),K),20000) ; T_trial+=1024) 
			check_log(filename, M, N, K, T_trial, A_loc, B_loc, C_loc, dev_id);
	}

	/// Matrix Layouts for GPU GEMM
	cublasOperation_t gpu_op_A = CUBLAS_OP_N, gpu_op_B = CUBLAS_OP_N;  // CUBLAS_OP_N, CUBLAS_OP_T

	size_t ldA = M, ldB = K, ldC = M;

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
#endif

	size_t cublasXt_tile;
	if (Tin > 0) cublasXt_tile = Tin;
	else cublasXt_tile = (size_t) fmin(fmin(M,N),K)/2; 

	// First call for Validate and/or additional overhead counting
	cpu_timer = csecond();
	cublasXt_sgemm_wrapper(gpu_op_A,  gpu_op_B, M, N, K, alpha, A, ldA, B, ldB, beta, C, ldC, cublasXt_tile, 0, dev_id);
	cudaCheckErrors();
	cpu_timer  = csecond() - cpu_timer;
	fprintf(stderr,"First CUBLASXT DGEMM call-> M = %zu, N = %zu, K = %zu, T = %zu\n", M, N, K, (size_t) cublasXt_tile);
	fprintf(stderr, "Total time:\t%lf ms\n\n", cpu_timer  * 1000);
	double first_over_t = cpu_timer; 

	// Half problem size tile
	cpu_timer = csecond();
	cublasXt_sgemm_wrapper(gpu_op_A,  gpu_op_B, M, N, K, alpha, A, ldA, B, ldB, beta, C, ldC, cublasXt_tile, 0, dev_id);
	cudaCheckErrors();
	cpu_timer = csecond() - cpu_timer;

	double cublasXt_t = cpu_timer; 
	if (Tin > 0){
		fprintf(stderr,"Running CUBLASXT SGEMM-> M = %zu, N = %zu, K = %zu, T = %zu\n", M, N, K, cublasXt_tile);
		cpu_timer  = csecond();
		cublasXt_sgemm_wrapper(gpu_op_A,  gpu_op_B, M, N, K, alpha, A, ldA, B, ldB, beta, C, ldC, cublasXt_tile, 0, dev_id);
		cudaCheckErrors();
		cpu_timer  = csecond() - cpu_timer;
		fprintf(stderr, "Total time:\t%lf ms\n", cpu_timer  * 1000);
		if (cublasXt_t > cpu_timer) cublasXt_t = cpu_timer; 
	}
	else {
		for (size_t T_trial = 1024; T_trial < 20000; T_trial+=1024) if (M >= T_trial*1.5 && N >= T_trial*1.5 && K >= T_trial*1.5){
			fprintf(stderr,"Running CUBLASXT SGEMM-> M = %zu, N = %zu, K = %zu, T = %zu\n", M, N, K, T_trial);
			cpu_timer  = csecond();
			cublasXt_sgemm_wrapper(gpu_op_A,  gpu_op_B, M, N, K, alpha, A, ldA, B, ldB, beta, C, ldC, T_trial, 0, dev_id);
			cudaCheckErrors();
			cpu_timer  = csecond() - cpu_timer;
			fprintf(stderr, "Total time:\t%lf ms\n", cpu_timer  * 1000);
			if (cpu_timer < cublasXt_t){
				cublasXt_t = cpu_timer;
				cublasXt_tile = T_trial;
			}
		}
		fprintf(stderr, "\nCUBLASXT SGEMM T_best = %zu : t = %lf ms ( %lf Gflops/s )\n\n", cublasXt_tile, cublasXt_t  * 1000, Gval_per_s(dgemm_flops(M,N,K),cublasXt_t));
	}

	fprintf(stderr,"Running CUBLASXT SGEMM-> M = %zu, N = %zu, K = %zu T_best = %zu\n", M, N, K, cublasXt_tile);
	double min_t = first_over_t, max_t = 0, avg_t = 0;
	cpu_timer = csecond();
	short bench_it = 100;
	if ( M >= 8192 || N >= 8192 || K >= 8192) bench_it = 10; 
	for(int it = 0; it < bench_it; it++){
		cpu_timer = csecond();
		cublasXt_sgemm_wrapper(gpu_op_A,  gpu_op_B, M, N, K, alpha, A, ldA, B, ldB, beta, C, ldC, cublasXt_tile, 0, dev_id);
		cudaCheckErrors();
		cpu_timer = csecond() - cpu_timer;
		if ( cpu_timer < min_t ) min_t = cpu_timer;
		if ( cpu_timer > max_t ) max_t = cpu_timer;
		avg_t += cpu_timer;
	}
	avg_t/=bench_it;
	fprintf(stderr, "cuBLASXt :\n\tavg_t = %lf ms ( %lf Gflops/s )\n\tmin_t = %lf ms ( %lf Gflops/s )\n\tmax_t = %lf ms ( %lf Gflops/s )\n", 
	avg_t  * 1000, Gval_per_s(dgemm_flops(M,N,K),avg_t),
	min_t  * 1000, Gval_per_s(dgemm_flops(M,N,K),min_t),
	max_t  * 1000, Gval_per_s(dgemm_flops(M,N,K),max_t));
	
	check_log(filename, M, N, K, cublasXt_tile, A_loc, B_loc, C_loc, dev_id);
	report_results(filename, M, N, K, cublasXt_tile, A_loc, B_loc, C_loc, dev_id, avg_t, min_t, max_t, first_over_t); 

	return 0;
}
