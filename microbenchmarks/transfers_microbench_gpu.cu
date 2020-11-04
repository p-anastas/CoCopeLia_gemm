///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief A transfer micro-benchmark from->to for a) contiguous transfers, b) non-cont square transfers, c) full bidirectional overlapped transfers 
///

#include <unistd.h>
#include <cassert>
#include <cuda.h>
#include "cublas_v2.h"
#include "cpu_utils.hpp"
#include "gpu_utils.hpp"
#include "testing.hpp"
#include "CoCopeLia.hpp"

void report_run(char* filename, size_t dim, double t_sq_av, double t_sq_min, double t_sq_max, double t_sq_bid_av, double t_sq_bid_min, double t_sq_bid_max){

	FILE* fp = fopen(filename,"a");
	if (!fp) error("report_run: LogFile failed to open");
   	fprintf(fp,"%d, %e,%e,%e,%e,%e,%e\n", dim, t_sq_av, t_sq_min, t_sq_max, t_sq_bid_av, t_sq_bid_min, t_sq_bid_max);
        fclose(fp); 
}

int main(const int argc, const char *argv[]) {

  	int ctr = 1, samples, dev_id, dev_count;

	char machine[256];
	short from, to; 
	size_t minDim = 0, maxDim = 0, step = 0; 

	switch (argc) {
	case (7):
		sprintf(machine , "%s", argv[ctr++]);
		to = atoi(argv[ctr++]);
		from = atoi(argv[ctr++]);
		minDim = atoi(argv[ctr++]);
		maxDim = atoi(argv[ctr++]);
		step = atoi(argv[ctr++]);
		break;
	default:
		error("Incorrect input arguments. Usage: ./correct_run machine to from minDim maxDim step");
  	}

	if (strcmp(MACHINE, machine)) error("transfers_microbench_gpu: Running on wrong machine");
	char *filename = (char *) malloc(256* sizeof(char));
	sprintf(filename, "%s/BenchOutputs/%s/cublasSet_Get_to-%d_from-%d_min-%d_max-%d_step-%d_iter-%d.log", PROJECTDIR, MACHINE, to, from, minDim, maxDim, step, ITER);
	check_benchmark(filename);


	fprintf(stderr,"\nTransfer benchmark@%s %s->%s : (%d,%d) with step %d\n", MACHINE, print_loc(from), print_loc(to), minDim, maxDim, step);

	cudaGetDeviceCount(&dev_count);

	if (minDim < 1) error("Transfer Microbench: Bytes must be > 0"); 
	else if ( dev_count < from + 1) error("Transfer Microbench: Src device does not exist"); 
	else if ( dev_count < to + 1) error("Transfer Microbench: Dest device does not exist"); 

	void* src, *dest, *rev_src, *rev_dest; 

	//Only model pinned memory transfers from host to dev and visa versa
  	if (from < 0 && to < 0) error("Transfer Microbench: Both locations are in host");
  	else if ( from >= 0 && to >= 0) error("Transfer Microbench: Both locations are devices - device communication not implemented");
	else if (from == -2 || to == -2) error("Transfer Microbench: Not pinned memory (synchronous)");
	
	size_t ldsrc, ldest = ldsrc = maxDim + 1; 

	vec_alloc(&src, maxDim*(maxDim+1)*8, from);
	vec_alloc(&dest, maxDim*(maxDim+1)*8, to);
	vec_alloc(&rev_src, maxDim*(maxDim+1)*8, to);
	vec_alloc(&rev_dest, maxDim*(maxDim+1)*8, from);

	/// Local Timers 
	double cpu_timer, t_sq_av, t_sq_min, t_sq_max, t_sq_bid_av, t_sq_bid_min, t_sq_bid_max;
	gpu_timer_p cuda_timer = gpu_timer_init();

	if (from < 0){
		Dvec_init_CoCoRand((double*)src, maxDim*(maxDim+1), 42, 1, 0);
		Dvec_init_CoCoRand((double*)rev_src, maxDim*(maxDim+1), 43, 0, to);
	}
	else {
		Dvec_init_CoCoRand((double*)src, maxDim*(maxDim+1), 42, 0, from);
		Dvec_init_CoCoRand((double*)rev_src, maxDim*(maxDim+1), 43, 1, 0);
	}

	cudaStream_t stream, reverse_stream;
	cudaStreamCreate(&stream);
	cudaStreamCreate(&reverse_stream);
	
	fprintf(stderr, "Warming up...\n");
	/// Warmup.
	for (int it = 0; it < 10; it++) {
		if(from == -1) cublasSetMatrixAsync(maxDim*(maxDim+1), 1, sizeof(double), src, maxDim*(maxDim+1), dest, maxDim*(maxDim+1),stream);
		else cublasGetMatrixAsync(maxDim*(maxDim+1), 1, sizeof(double), src, maxDim*(maxDim+1), dest, maxDim*(maxDim+1),stream);
	}
	cudaCheckErrors();

	for (size_t dim = minDim; dim < maxDim+1; dim+=step){
		t_sq_av = t_sq_max = t_sq_bid_av = t_sq_bid_max = 0;
		t_sq_min = t_sq_bid_min = 1e9; 
		fprintf(stderr, "Cublas-chunk Link %s->%s (Chunk %dx%d):\n", print_loc(from), print_loc(to), dim, dim);
		for (int it = 0; it < ITER ; it++) {
			cpu_timer = - csecond();
			if(from == -1) cublasSetMatrixAsync(dim, dim, 8, src, ldsrc, dest, ldest, stream);
			else cublasGetMatrixAsync(dim, dim, 8, src, ldsrc, dest, ldest, stream);
			cudaStreamSynchronize(stream);
			cpu_timer = csecond() + cpu_timer;
			t_sq_av += cpu_timer;
			if (cpu_timer > t_sq_max) t_sq_max = cpu_timer; 
			if (cpu_timer < t_sq_min) t_sq_min = cpu_timer; 
		}
		cudaCheckErrors();
		t_sq_av = t_sq_av/ITER;
		fprintf(stderr, "Transfer time:\t Average=%lf ms ( %lf Gb/s), Min = %lf ms, Max = %lf ms\n", t_sq_av  * 1000, Gval_per_s(dim*dim*8, t_sq_av), t_sq_min  * 1000, t_sq_max  * 1000);

		fprintf(stderr, "Reverse overlapped Link %s->%s (Chunk %dx%d):\n", print_loc(from), print_loc(to), dim, dim);
		for (int it = 0; it < ITER ; it++) {
			for (int rep = 0; rep < 10 ; rep++) {
				if(to == -1) cublasSetMatrixAsync(dim, dim, 8, rev_src, ldsrc, rev_dest, ldest, reverse_stream);
				else cublasGetMatrixAsync(dim, dim, 8, rev_src, ldsrc, rev_dest, ldest, reverse_stream);
			}
			gpu_timer_start(cuda_timer, stream);
			if(from == -1) cublasSetMatrixAsync(dim, dim, 8, src, ldsrc, dest, ldest,stream);
			else cublasGetMatrixAsync(dim, dim, 8, src, ldsrc, dest, ldest,stream);
			gpu_timer_stop(cuda_timer, stream);
			cudaCheckErrors();
			t_sq_bid_av += gpu_timer_get(cuda_timer);
			if (gpu_timer_get(cuda_timer) > t_sq_bid_max) t_sq_bid_max = gpu_timer_get(cuda_timer); 
			if (gpu_timer_get(cuda_timer) < t_sq_bid_min) t_sq_bid_min = gpu_timer_get(cuda_timer); 
		}
		cudaCheckErrors();
		t_sq_bid_av = t_sq_bid_av/ITER/1000;
		t_sq_bid_min/= 1000;
		t_sq_bid_max/= 1000;
		fprintf(stderr, "Transfer time:\t Average=%lf ms ( %lf Gb/s), Min = %lf ms, Max = %lf ms\n", t_sq_bid_av  * 1000, Gval_per_s(dim*dim*8, t_sq_bid_av), t_sq_bid_min  * 1000, t_sq_bid_max  * 1000);
		
		report_run(filename, dim, t_sq_av, t_sq_min, t_sq_max, t_sq_bid_av, t_sq_bid_min, t_sq_bid_max);

	}
	vec_free(&src, from);
	vec_free(&dest, to); 
	vec_free(&rev_src, to);
	vec_free(&rev_dest, from); 
	return 0;
}
