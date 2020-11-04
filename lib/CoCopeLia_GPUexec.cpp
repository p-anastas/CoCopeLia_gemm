///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief The execution lookup functions for CoCopeLia.
///

#include <stdlib.h>
#include <math.h>

#include "cpu_utils.hpp"
#include "CoCopeLia_GPUexec.hpp"

GPUexec3Model_p GPUexec3Model_init(short dev_id, char* func, short mode){
	if (strcmp(func, "Dgemm") && strcmp(func, "Sgemm") ) error("GPUexec3Model_init: Invalid/Not implemented func");
	if (mode < 1 || mode > 3) error("GPUexec3Model_init: Unknown mode");
	GPUexec3Model_p out_model = (GPUexec3Model_p) malloc(sizeof(struct  BLAS3_data));
	char filename[256];
	sprintf(filename, "%s/BenchOutputs/%s/cublas%s_dev-%d_min-%d_max-%d_step-%d_iter-%d.log", PROJECTDIR, MACHINE, func, dev_id, minDim_blas3, maxDim_blas3, step_blas3, ITER);
	FILE* fp = fopen(filename,"r");
	if (!fp) {
		fprintf(stderr, "GPUexec3Model_init: Logfile = %s\n", filename);
		error("GPUexec3Model_init: t_exec3 LogFile not generated");
	}
	size_t bench_lines = (maxDim_blas3 - minDim_blas3)/step_blas3 + 1;
	fprintf(stderr, "GPUexec3Model_init: Reading %zu lines from %s\n", bench_lines, filename);
	int items;
	size_t trashdata, chech_tile = minDim_blas3; 
	for (int i = 0; i < bench_lines; i++){
		items = fscanf(fp, "%zu,%zu,%zu, %lf,%lf,%lf\n", &trashdata, &trashdata, &trashdata, &out_model->av_time_buffer[i], &out_model->min_time_buffer[i], &out_model->max_time_buffer[i]);
		if (items != 6) error("GPUexec3Model_init_inter: Problem in reading model");
		if (trashdata != chech_tile) error("GPUexec3Model_init_inter: Some tile was skipped in benchmarks");
		//fprintf(stderr, "GPUexec3Model_init: Scanned entry %d: T = %zu -> t_av = %lf ms, t_min = %lf ms, t_max = %lf ms\n", i, chech_tile, out_model->av_time_buffer[i]*1000, out_model->min_time_buffer[i]*1000, out_model->max_time_buffer[i]*1000);
		chech_tile+=step_blas3;
    	}

	out_model->mode = mode; 
	out_model->dev_id = dev_id; 
	out_model->func = func; 
	return out_model; 
}

double GPUexec3Model_predict(GPUexec3Model_p model, size_t T){
	double result = 0; 
	if( !T ) return result;  
	//if( T < 1024 ) warning("GPUexec3Model_predict: small dim in prediction");
	if (T < minDim_blas3) error("GPUexec3Model_predict: Tile (T) smaller than micro-benchmark min");
	else if (T > maxDim_blas3) error("GPUexec3Model_predict: Tile (T) larger than micro-benchmark max");
	else if ( (T - minDim_blas3)%step_blas3 != 0) error("GPUexec3Model_predict: Tile (T) not part of micro-benchmark");
	size_t offset = (T - minDim_blas3)/step_blas3;
	if (model->mode == 1) result = model->av_time_buffer[offset];
	else if (model->mode == 2) result = model->min_time_buffer[offset];
	else if (model->mode == 3) result = model->max_time_buffer[offset];
	else error("GPUexec3Model_predict: Uninialized model");
	return result; 
}
