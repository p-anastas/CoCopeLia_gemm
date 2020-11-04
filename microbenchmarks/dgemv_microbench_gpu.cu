///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief A cublasDgemv micro-benchmark
///

#include <cassert>
#include "cpu_utils.hpp"
#include "gpu_utils.hpp"
#include "testing.hpp"

void report_run(char* filename, short dev_id, size_t M, size_t N, double cublas_t_av, double cublas_t_min, double cublas_t_max){

	FILE* fp = fopen(filename,"a");
	if (!fp) error("report_run: LogFile failed to open");
   	fprintf(fp,"%d,%d, %e,%e,%e\n", M, N, cublas_t_av, cublas_t_min, cublas_t_max);
        fclose(fp); 
}

int main(const int argc, const char *argv[]) {

  	double alpha, beta;
  	alpha = 1.1234, beta = 1.2345;

  	int ctr = 1, dev_id, bench_num = 0;

	char machine[256];
	size_t minDim, maxDim;
	size_t step;
	size_t incx = 1, incy = 1;

	switch (argc) {
	case (6):
		sprintf(machine , "%s", argv[ctr++]);
		dev_id = atoi(argv[ctr++]);
		minDim = atoi(argv[ctr++]);
		maxDim = atoi(argv[ctr++]);
		step = atoi(argv[ctr++]);
		break;
	default:
		error("Incorrect input arguments. Usage: ./correct_run machine dev_id minDim maxDim maxDim step step max_benchmarks");
  	}

	if (strcmp(MACHINE, machine)) error("dgemv_microbench_gpu: Running on wrong machine");
	char *filename = (char *) malloc(256* sizeof(char));
	sprintf(filename, "%s/BenchOutputs/%s/cublasDgemv_dev-%d_min-%d_max-%d_step-%d_iter-%d.log", PROJECTDIR, MACHINE, dev_id, minDim, maxDim, step, ITER);
	check_benchmark(filename);

	/// Matrix Layouts for GPU GEMM
	cublasOperation_t gpu_op_A = CUBLAS_OP_N;  // CUBLAS_OP_N, CUBLAS_OP_T

	size_t ldA = maxDim;

	/// Set device 
	cudaSetDevice(dev_id);

	cublasHandle_t handle0;
 	cudaStream_t host_stream;

  	cudaStreamCreate(&host_stream);
	assert(CUBLAS_STATUS_SUCCESS == cublasCreate(&handle0));
	assert(CUBLAS_STATUS_SUCCESS == cublasSetStream(handle0, host_stream));

	fprintf(stderr, "\nAllocating device memory...");
	double cpu_timer = csecond();

	double *A_dev, *x_dev, *y_dev;
  	vec_alloc((void**)&A_dev, maxDim * maxDim * sizeof(double), dev_id);
  	vec_alloc((void**)&x_dev, maxDim * sizeof(double), dev_id);
  	vec_alloc((void**)&y_dev, maxDim * sizeof(double), dev_id);
	cudaCheckErrors();

	cpu_timer  = csecond() - cpu_timer ;
	fprintf(stderr, "done.\nAlloc time:\t%lf ms\n\n",  cpu_timer  * 1000);

	fprintf(stderr, "Initializing to random values..."); 
	cpu_timer = csecond();

	Dvec_init_cuRAND(A_dev, maxDim * maxDim, 42);
	Dvec_init_cuRAND(x_dev, maxDim, 43);
	Dvec_init_cuRAND(y_dev, maxDim, 44);

	cudaCheckErrors();
	cpu_timer  = csecond() - cpu_timer ;	
	fprintf(stderr, "done.\nInit time:\t%lf ms\n\n",  cpu_timer  * 1000);

	fprintf(stderr, "\nTile details: A(%s) x(inc=%d) y(inc=%d) -> maxDim = %d, maxDim = %d\n",
            print_mem(COL_MAJOR), 1, 1, maxDim, maxDim);
	fprintf(stderr, "Constants: alpha = %lf, beta = %lf\n", alpha, beta);

	// Warmup 
	for ( int itt = 0; itt <10; itt++){
		assert(CUBLAS_STATUS_SUCCESS == cublasDgemv(handle0, gpu_op_A, maxDim, maxDim, &alpha, A_dev, ldA, x_dev, incx, &beta, y_dev, incy));
		cudaStreamSynchronize(host_stream);
	}
	cudaCheckErrors();
	double cublas_t_av, cublas_t_min , cublas_t_max; 
	size_t bench_ctr = 0;
	for (size_t M = minDim; M < maxDim + 1; M+=step){
			size_t N = maxDim;
			fprintf(stderr,"Running cublasDgemv-> M = %d, N = %d:\n", M, N);
			cublas_t_av = cublas_t_max = 0;
			cublas_t_min = 1e9;
			for (int itt = 0; itt < ITER; itt ++) {
				cpu_timer = csecond();
				assert(CUBLAS_STATUS_SUCCESS == cublasDgemv(handle0, gpu_op_A, M, N, &alpha, A_dev, ldA, x_dev, incx, &beta, y_dev, incy));
				cudaStreamSynchronize(host_stream);
				cpu_timer  = csecond() - cpu_timer ;
				cublas_t_av += cpu_timer;
				if (cpu_timer > cublas_t_max) cublas_t_max = cpu_timer; 
				if (cpu_timer < cublas_t_min) cublas_t_min = cpu_timer; 
			}
			cublas_t_av /= ITER;
			fprintf(stderr, "GPU exec time:\t Average=%lf ms, Min = %lf ms, Max = %lf ms\n", cublas_t_av  * 1000, cublas_t_min  * 1000, cublas_t_max  * 1000);
			cudaCheckErrors();

			report_run(filename, dev_id, M, N, cublas_t_av, cublas_t_min, cublas_t_max); 
			bench_ctr++;
	}
	fprintf(stderr, "Ran %d Benchmarks.Finallizing...\n", bench_ctr);
	return 0;
}
