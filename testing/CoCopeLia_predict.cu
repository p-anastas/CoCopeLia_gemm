///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief The start of Zawarudo
///

#define maxDim 16384
#define minDim 512
#define step 256

#include <cassert>
#include <cuda.h>
#include <cblas.h>
#include "cublas_v2.h"
#include "cpu_utils.hpp"
#include "gpu_utils.hpp"
#include "CoCopeLia.hpp"
#include "CoCopeLia_front.hpp"

int main(const int argc, const char *argv[]) {

  	int ctr = 1;

	char machine[256], func[256];
	size_t M, N, K;

	int dev_id = -1; 

	short first_loc = 1, second_loc = 1, third_loc = 1; 
	switch (argc) {
	case (10):
		first_loc = atoi(argv[argc-3]);
		second_loc = atoi(argv[argc-2]);	
		// Irrelevant for AXPY
		third_loc = atoi(argv[argc-1]);
	case (7):
		sprintf(machine , "%s", argv[ctr++]);
		M = atoi(argv[ctr++]);
		// Irrelevant for AXPY
		N = atoi(argv[ctr++]);
		// Irrelevant for AXPY, GEMV
		K = atoi(argv[ctr++]);
		dev_id = atoi(argv[ctr++]);
		sprintf(func , "%s", argv[ctr++]);
		break;
	default:
		error("Incorrect input arguments. Usage: ./correct_run machine M N K dev_id func(={dgemm,sgemm,dgemv,daxpy}) [A_loc B_loc C_loc]");
  	}

	if (strcmp(MACHINE, machine)) error("CoCopelia_predict: Running on wrong machine");

	fprintf(stderr, "\nInput details: -> M = %d, N = %d, K = %d\n",M, N, K);

	//log_CoCopelia_prediction(M, N, K, fmin(maxDim,fmin(fmin(M,N),K))/4, first_loc, second_loc, third_loc, dev_id, func, 1);
	for (size_t Ts = minDim; Ts <= (size_t) min((size_t)maxDim,min(min(M,N),K)) ; Ts +=step) log_CoCopelia_prediction(M, N, K, Ts, first_loc, second_loc, third_loc, dev_id, func, 1);
	for (size_t Ts = minDim; Ts <= (size_t) min((size_t)maxDim,min(min(M,N),K)); Ts +=step) log_CoCopelia_prediction(M, N, K, Ts,  first_loc, second_loc, third_loc, dev_id, func, 2);
	for (size_t Ts = minDim; Ts <= (size_t) min((size_t)maxDim,min(min(M,N),K)); Ts +=step) log_CoCopelia_prediction(M, N, K, Ts, first_loc, second_loc, third_loc, dev_id, func, 3);
	for (size_t Ts = minDim; Ts <= (size_t) min((size_t)maxDim,min(min(M,N),K)); Ts +=step) log_CoCopelia_prediction(M, N, K, Ts, first_loc, second_loc, third_loc, dev_id, func, 4);
	return 0;
}
