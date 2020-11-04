#ifndef COCOPELIAGPUEXEC_H
#define COCOPELIAGPUEXEC_H

#define maxDim_blas3 16384
#define minDim_blas3 256
#define step_blas3 256

#define maxDim_blas2 16384
#define minDim_blas2 256
#define step_blas2 256

#define maxDim_blas1 268435456
#define minDim_blas1 1048576
#define N_step_blas1 1048576

typedef struct interpolation_val{
    size_t M, N, K;
    double time;
} * inter_p;

typedef struct  BLAS1_data{
	// 0 for undef, 1 for linear regression, 2 for interpolation
	short pred_flag;

	double inter;
	double a0;

	inter_p * inter_data;
	size_t inter_data_num;

	short dev_id; 
	char* machine, * func;
}* GPUexec1Model_p;

typedef struct  BLAS2_data{
	// 0 for undef, 1 for linear regression, 2 for interpolation
	short pred_flag;

	double inter;
	double a0, a1;
	double b0;

	inter_p * inter_data;
	size_t inter_data_num;

	short dev_id; 
	char* machine, * func;
}* GPUexec2Model_p;

typedef struct  BLAS3_data{
	short dev_id; 
	char* func;
	
	// 0 for undef, 1 for average, TODO: 2 for min, 3 for max
	short mode;
	double av_time_buffer[(maxDim_blas3-minDim_blas3)/step_blas3 + 1];
	// TODO: These can be used for more robust results or for worst/best case performance prediction
	double min_time_buffer[(maxDim_blas3-minDim_blas3)/step_blas3 + 1];
	double max_time_buffer[(maxDim_blas3-minDim_blas3)/step_blas3 + 1];
}* GPUexec3Model_p;


/// Load parameters from file and return  BLAS 1 execution time model
GPUexec1Model_p GPUexec1Model_init(short dev,  char* func, short mode);

/// Load parameters from file and return  BLAS 3 execution time model
double GPUexec1Model_predict(GPUexec1Model_p model, size_t D1);

/// Load parameters from file and return  BLAS 2 execution time model
GPUexec2Model_p GPUexec2Model_init(short dev,  char* func, short mode);

/// Load parameters from file and return  BLAS 3 execution time model
double GPUexec2Model_predict(GPUexec2Model_p model, size_t D1,  size_t D2);

/// Load parameters from file and return  BLAS 3 execution time model
GPUexec3Model_p GPUexec3Model_init(short dev,  char* func, short mode);

/// Load parameters from file and return  BLAS 3 execution time model
double GPUexec3Model_predict(GPUexec3Model_p model, size_t T);

#endif
