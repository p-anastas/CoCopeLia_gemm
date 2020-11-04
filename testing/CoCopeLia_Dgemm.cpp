///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief The start of Zawarudo
///

#include "cpu_utils.hpp"
#include "CoCopeLia.hpp"
#include "testing.hpp"
// TODO: Preliminary predict tile in order to avoid repeating benchmarks. NOT required.
#include "CoCopeLia_Model.hpp"

int main(const int argc, const char *argv[]) {

  	double alpha, beta;
  	alpha = 1.1234, beta = 1.2345;

  	int ctr = 1;

	char machine[256];
	size_t M = 0, N = 0, K = 0;

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

	if (strcmp(MACHINE, machine)) error("CoCopeLia_Dgemm: Running on wrong machine");
	else fprintf(stderr, "Problem configuration (M=%zu, N=%zu, K=%zu) on locations (A_loc=%d, B_loc=%d, C_loc=%d)\n", M, N, K, A_loc, B_loc, C_loc);

	char *filename = (char *) malloc(256* sizeof(char));
	if (Tin > 0) {
		if ( Tin > M || Tin > N || Tin > K) error("Given Tin bigger than problem dim");
		else if (Tin > M/1.5 && Tin > N/1.5 && Tin > K/1.5) error("Given Tin bigger than all problem dims/1.5");
		sprintf(filename, "%s/Data_manipulation/Results/%s/validation/CoCopeLia_Dgemm_%d_v%s.log", PROJECTDIR, MACHINE, dev_id, VERSION);
		check_log(filename, M, N, K, Tin, A_loc, B_loc, C_loc, dev_id);
	}
	else {
		sprintf(filename, "%s/Data_manipulation/Results/%s/evaluation/CoCopeLia_Dgemm_%d_v%s.log", PROJECTDIR, MACHINE, dev_id, VERSION);
		// TODO: Preliminary predict tile in order to avoid repeating benchmarks. NOT required.
		CoCoModel_p model = CoCoModel_gemm_init( M, N, K, A_loc, B_loc, C_loc, dev_id, "Dgemm", 1);
		size_t T = CoCoModel_optimize3(model);
		check_log(filename, M, N, K, T, A_loc, B_loc, C_loc, dev_id);
	}

	/// Matrix Layouts for CPU GEMM
	CBLAS_TRANSPOSE cpu_op_A = CblasNoTrans, cpu_op_B = CblasNoTrans;    // CblasNoTrans, CblasTrans
	CBLAS_LAYOUT cblas_layout = CblasColMajor;

	size_t ldA = M, ldB = K, ldC = M;

	/// Local Timers 
	double cpu_timer = csecond();

	fprintf(stderr, "\nAllocating memory...");

	double *A, *B, *C;
	// allocate in device if loc = 0, otherwise allocate in pinned memory for benchmarks
	A = (double*) CoComalloc(M * K*sizeof(double), A_loc, dev_id);
	B = (double*) CoComalloc(N * K*sizeof(double), B_loc, dev_id);
	C = (double*) CoComalloc(M * N*sizeof(double), C_loc, dev_id);

	cudaCheckErrors();
	cpu_timer  = csecond() - cpu_timer;
	fprintf(stderr, "done.\nAlloc time:\t%lf ms\n\n",  cpu_timer  * 1000);

#ifdef VALIDATE
	cpu_timer = csecond();
	fprintf(stderr, "Initializing to random values (VALIDATE)..."); 
	Dvec_init_CoCoRand(A, K * M, 42, A_loc, dev_id);
	Dvec_init_CoCoRand(B, K * N, 43, B_loc, dev_id);
	Dvec_init_CoCoRand(C, M * N, 44, C_loc, dev_id);
	cudaCheckErrors();
	cpu_timer  = csecond() - cpu_timer ;
	fprintf(stderr, "done.\nInit time:\t%lf ms\n\n",  cpu_timer  * 1000);

	/// Matrix Layouts for GPU GEMM
	cublasOperation_t gpu_op_A = CUBLAS_OP_N, gpu_op_B = CUBLAS_OP_N;  // CUBLAS_OP_N, CUBLAS_OP_T
	double *C_out, *C_out1;
	C_out  = (double*) malloc(M * N*sizeof(double));
	C_out1  = (double*) malloc(M * N*sizeof(double));

 	if (C_loc) optimized_memcpy(C_out1, C,  M * N *sizeof(double));
 	else vec_get_memcpy(C_out1, C,  M * N *sizeof(double), dev_id);
#endif

	// First call for Validate and/or additional overhead counting
	cpu_timer = csecond();
	CoCopeLia_Dgemm(cpu_op_A, cpu_op_B, M, N, K, alpha, A, ldA, B, ldB, beta, C , ldC, &Tin, dev_id);
	cudaCheckErrors();
	cpu_timer  = csecond() - cpu_timer;
	fprintf(stderr,"First CoCopelia DGEMM call-> M = %zu, N = %zu, K = %zu\n", M, N, K);
	fprintf(stderr, "Total time:\t%lf ms\n\n", cpu_timer  * 1000);
	double first_over_t = cpu_timer; 

#ifdef VALIDATE
 	if (C_loc) optimized_memcpy(C_out, C,  M * N *sizeof(double));
 	else vec_get_memcpy(C_out, C,  M * N *sizeof(double), dev_id);
#endif

	fprintf(stderr,"Running CoCopeLia DGEMM-> M = %zu, N = %zu, K = %zu\n", M, N, K);
	double min_t = first_over_t, max_t = 0, avg_t = 0;
	cpu_timer = csecond();
	short bench_it = 100;
	if ( M >= 8192 || N >= 8192 || K >= 8192) bench_it = 10; 
	for(int it = 0; it < bench_it; it++){
		cpu_timer = csecond();
		CoCopeLia_Dgemm(cpu_op_A, cpu_op_B, M, N, K, alpha, A, ldA, B, ldB, beta, C , ldC, &Tin, dev_id);
		cudaCheckErrors();
		cpu_timer = csecond() - cpu_timer;
		if ( cpu_timer < min_t ) min_t = cpu_timer;
		if ( cpu_timer > max_t ) max_t = cpu_timer;
		avg_t += cpu_timer;
	}
	avg_t/=bench_it;
	fprintf(stderr, "\nCoCopeLia :\n\tavg_t = %lf ms ( %lf Gflops/s )\n\tmin_t = %lf ms ( %lf Gflops/s )\n\tmax_t = %lf ms ( %lf Gflops/s )\n", 
	avg_t  * 1000, Gval_per_s(dgemm_flops(M,N,K),avg_t),
	min_t  * 1000, Gval_per_s(dgemm_flops(M,N,K),min_t),
	max_t  * 1000, Gval_per_s(dgemm_flops(M,N,K),max_t));

#ifdef VALIDATE
 	if (C_loc) optimized_memcpy(C, C_out1,  M * N *sizeof(double));
 	else vec_set_memcpy(C, C_out1,  M * N *sizeof(double), dev_id);
	cublasXt_dgemm_wrapper(gpu_op_A,  gpu_op_B, M, N, K, alpha, A, ldA, B, ldB, beta, C, ldC,  (size_t) fmin(fmin(M,N),K)/2/2, 0, dev_id);
 	if (C_loc) optimized_memcpy(C_out1, C,  M * N *sizeof(double));
	else vec_get_memcpy(C_out1, C,  M * N *sizeof(double), dev_id);
 	if(Dtest_equality(C_out1, C_out, M * N) < 5) error("Insufficient accuracy for benchmarks");
#endif

	check_log(filename, M, N, K, Tin, A_loc, B_loc, C_loc, dev_id);
	report_results(filename, M, N, K, Tin, A_loc, B_loc, C_loc, dev_id, avg_t, min_t, max_t, first_over_t); 
	cudaCheckErrors();

	return 0;
}
