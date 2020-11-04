///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief Some CUDA function calls with added error-checking
///

#include <stdlib.h>
#include <math.h>

#include "cpu_utils.hpp"
#include "CoCopeLia_GPUexec.hpp"



GPUexec1Model_p GPUexec1Model_init_lr(short dev_id, char* func){
	GPUexec1Model_p out_model = (GPUexec1Model_p) malloc(sizeof(struct  BLAS1_data));
	char filename[256];
	sprintf(filename, "%s/Data_manipulation/Results/%s/Models/%s_model_%d.log", PROJECTDIR, MACHINE, func, dev_id);
	FILE* fp = fopen(filename,"r");
	if (!fp) {
		fprintf(stderr, "GPUexec1Model_init: Logfile = %s\n", filename);
		error("GPUexec1Model_init: t_exec1 LogFile not generated");
	}
	int items = fscanf(fp, "%lf\n%lf\n", &out_model->inter, &out_model->a0);
	if (items != 2) error("GPUexec1Model_init_lr: Problem in reading model");
	fclose(fp);
	out_model->dev_id = dev_id; 
	out_model->machine = MACHINE; 
	out_model->func = func; 
	out_model->pred_flag = 1; 
	out_model->inter_data = NULL;
	out_model->inter_data_num =  0;
	fprintf(stderr, "GPUexec1Model_init : t_%s(dev=%d) model initialized for %s ->\ninter = %e\na0 = %e\n", out_model->func, out_model->dev_id, out_model->machine, out_model->inter, out_model->a0);
	return out_model;
}

/*
GPUexec1Model_p GPUexec1Model_init_inter(short dev_id, char* func){
	GPUexec1Model_p out_model = (GPUexec1Model_p) malloc(sizeof(struct  BLAS1_data));
	inter_p inter_data[sample_lim_blas1];

	char filename[256];
	sprintf(filename, "%s/Data_manipulation/Results/%s/microbenchmarks/%s_gpu_microbench_%d_max-%d_step-%d_lim-%d.log", PROJECTDIR, MACHINE, func, dev_id, Nmax_blas1, N_step_blas1, sample_lim_blas1);
	FILE* fp = fopen(filename,"r");
	if (!fp) {
		fprintf(stderr, "GPUexec1Model_init: Logfile = %s\n", filename);
		error("GPUexec1Model_init: t_exec1 LogFile not generated");
	}
	size_t bench_lines = fmin( sample_lim_blas1, (Nmax_blas1 - 512)/N_step_blas1 + 1);
	fprintf(stderr, "GPUexec1Model_init: Reading %zu lines from %s\n", bench_lines, filename);
	double trashdata;
	int items;
	for (int i = 0; i < bench_lines; i++){   
		inter_data[i] = (inter_p) malloc(sizeof(struct interpolation_val));
		items = fscanf(fp, "%zu, %lf,%lf,%lf\n", &inter_data[i]->N, &inter_data[i]->time, &trashdata, &trashdata);
		//if (items != 7) error("GPUexec3Model_init_inter: Problem in reading model");
		//fprintf(stderr, "GPUexec1Model_init: Scanned entry %d: {N} = {%zu} t = %lf ms\n", i, inter_data[i]->N, inter_data[i]->time*1000);
    	}
	out_model->inter_data = inter_data;
	out_model->inter_data_num =  bench_lines;
	out_model->pred_flag = 2; 
	out_model->dev_id = dev_id; 
	out_model->machine = MACHINE; 
	out_model->func = func; 

	return out_model;
}
*/

GPUexec1Model_p GPUexec1Model_init(short dev_id,  char* func, short mode)
{
	if (strcmp(func, "daxpy")) error("GPUexec1Model_init: Invalid/Not implemented func");
	if (mode == 1 || mode == 3) return GPUexec1Model_init_lr(dev_id, func);
	else if (mode == 2)  error("GPUexec1Model_init: Interpolation out of order");  //return GPUexec1Model_init_inter(dev_id, func);
	else error("Invalid/Not implemented exec mode");
}

double GPUexec1Model_predict(GPUexec1Model_p model, size_t D1)
{
	double result = 0; 
	if( !D1 ) return 0;  
	if( D1 < 256) warning("GPUexec1Model_predict: small dim in prediction");
	if (model->pred_flag == 1) {
		result = model->inter + model->a0 * D1;
		if (result > 0) return result;
		else return 1e9;
	}
	// FIXME: Only works for sorted microbench data
	else if (model->pred_flag == 2) {
		size_t Nup = 0 , Nlo = 0;
		for (int i = 0; i< model->inter_data_num; i++){
			//GEMM value match
			if ( D1 == model->inter_data[i]->N) return model->inter_data[i]->time;
			if (model->inter_data[i]->N <= D1) Nlo = model->inter_data[i]->N; 
			else if(!Nup) Nup = model->inter_data[i]->N; 
			if (Nup) break; 
		}
		warning("GPUexec1Model_predict: Performing questionable interpolation");
		if (!Nlo) return model->inter_data[0]->time*D1/Nup;
		else if (!Nup) return model->inter_data[0]->time*D1/Nlo;
		double t_lo = 0 , t_up = 0;
		for (int i = 0; i< model->inter_data_num; i++){
			//GEMM value match
			if (Nlo == model->inter_data[i]->N) t_lo =  model->inter_data[i]->time; 
			if (Nup == model->inter_data[i]->N ) t_up =  model->inter_data[i]->time;
		}
		//GEMM value match
		if (t_lo && t_up) return (t_lo*D1/Nlo + t_up*D1/Nup)/2;
		error("GPUexec1Model_predict: Failed on simple interpolation...");
	}
	else error("GPUexec1Model_predict: Uninialized model");
}

GPUexec2Model_p GPUexec2Model_init_lr(short dev_id, char* func){
	GPUexec2Model_p out_model = (GPUexec2Model_p) malloc(sizeof(struct  BLAS2_data));
	char filename[256];
	sprintf(filename, "%s/Data_manipulation/Results/%s/Models/%s_model_%d.log", PROJECTDIR, MACHINE, func, dev_id);
	FILE* fp = fopen(filename,"r");
	if (!fp) {
		fprintf(stderr, "GPUexec2Model_init: Logfile = %s\n", filename);
		error("GPUexec2Model_init: t_exec2 LogFile not generated");
	}
	int items = fscanf(fp, "%lf\n%lf\n%lf\n%lf\n", &out_model->inter, &out_model->a0, &out_model->a1, &out_model->b0);
	if (items != 4) error("GPUexec2Model_init_lr: Problem in reading model");
	fclose(fp);
	out_model->dev_id = dev_id; 
	out_model->machine = MACHINE; 
	out_model->func = func; 
	out_model->pred_flag = 1; 
	out_model->inter_data = NULL;
	out_model->inter_data_num =  0;
	fprintf(stderr, "GPUexec2Model_init : t_%s(dev=%d) model initialized for %s ->\ninter = %e\na0 = %e, a1 = %e\nb0 = %e\n", out_model->func, out_model->dev_id, out_model->machine, out_model->inter, out_model->a0, out_model->a1, out_model->b0);
	return out_model;
}

/*
GPUexec2Model_p GPUexec2Model_init_inter(short dev_id, char* func){
	GPUexec2Model_p out_model = (GPUexec2Model_p) malloc(sizeof(struct  BLAS2_data));
	inter_p inter_data[sample_lim_blas2];

	char filename[256];
	sprintf(filename, "%s/Data_manipulation/Results/%s/microbenchmarks/%s_gpu_microbench_%d_max-%d-%d_step-%d-%d_lim-%d.log", PROJECTDIR, MACHINE, func, dev_id, Mmax_blas2, Nmax_blas2, M_step_blas2, N_step_blas2, sample_lim_blas2);
	FILE* fp = fopen(filename,"r");
	if (!fp) {
		fprintf(stderr, "GPUexec2Model_init: Logfile = %s\n", filename);
		error("GPUexec2Model_init: t_exec2 LogFile not generated");
	}
	size_t bench_lines = fmin( sample_lim_blas2, ((Mmax_blas2 - 512)/M_step_blas2 + 1)*((Nmax_blas2 - 512)/N_step_blas2 + 1));
	fprintf(stderr, "GPUexec2Model_init: Reading %zu lines from %s\n", bench_lines, filename);
	double trashdata;
	int items;
	for (int i = 0; i < bench_lines; i++){   
		inter_data[i] = (inter_p) malloc(sizeof(struct interpolation_val));
		items = fscanf(fp, "%zu,%zu, %lf,%lf,%lf\n", &inter_data[i]->M, &inter_data[i]->N, &inter_data[i]->time, &trashdata, &trashdata);
		//if (items != 7) error("GPUexec3Model_init_inter: Problem in reading model");
		//fprintf(stderr, "GPUexec2Model_init: Scanned entry %d: {M,N} = {%zu,%zu} t = %lf ms\n", i, inter_data[i]->M, inter_data[i]->N, inter_data[i]->time*1000);
    	}
	out_model->inter_data = inter_data;
	out_model->inter_data_num =  bench_lines;
	out_model->pred_flag = 2; 
	out_model->dev_id = dev_id; 
	out_model->machine = MACHINE; 
	out_model->func = func; 

	return out_model;
}
*/

GPUexec2Model_p GPUexec2Model_init(short dev_id,  char* func, short mode){
	if (strcmp(func, "dgemv") ) error("GPUexec2Model_init: Invalid/Not implemented func");
	if (mode == 1 || mode == 3) return GPUexec2Model_init_lr(dev_id, func);
	else if (mode == 2)  error("GPUexec2Model_init: Interpolation out of order"); //return GPUexec2Model_init_inter(dev_id, func);
	else error("GPUexec2Model_init: Invalid/Not implemented exec mode");
}

double GPUexec2Model_predict(GPUexec2Model_p model, size_t D1,  size_t D2)
{
	double result = 0; 
	if( !D1 || !D2) return 0;  
	if( D1 < 256 || D2 < 256) warning("GPUexec2Model_predict: small dim in prediction");
	if (model->pred_flag == 1) {
		result = model->inter + model->a0 * D1  + model->a1 * D2 + model->b0 * D1 * D2;
		if (result > 0) return result;
		else return 1e9;
	}
	// FIXME: Only works for sorted microbench data
	else if (model->pred_flag == 2) {
		size_t Mup = 0, Mlo = 0 ,Nup = 0 , Nlo = 0;
		for (int i = 0; i< model->inter_data_num; i++){
			//GEMM value match
			if ( D1 == model->inter_data[i]->M && D2 == model->inter_data[i]->N) return model->inter_data[i]->time;
			if (model->inter_data[i]->M <= D1) Mlo = model->inter_data[i]->M; 
			else if(!Mup) Mup = model->inter_data[i]->M; 
			if (model->inter_data[i]->N <= D2) Nlo = model->inter_data[i]->N; 
			else if(!Nup) Nup = model->inter_data[i]->N;
			if ( Mup && Nup) break; 
		}
		warning("GPUexec2Model_predict: Performing questionable interpolation");	
		double t_lo = 0 , t_up = 0;
		for (int i = 0; i< model->inter_data_num; i++){
			//GEMM value match
			if ( Mlo == model->inter_data[i]->M && Nlo == model->inter_data[i]->N) {t_lo =  model->inter_data[i]->time; }
			if ( Mup == model->inter_data[i]->M && Nup == model->inter_data[i]->N) { t_up =  model->inter_data[i]->time; return (t_lo*D1/Mlo*D2/Nlo + t_up*D1/Mlo*D2/Nlo)/2; }
			else if ( Mup == model->inter_data[i]->M && Nlo == model->inter_data[i]->N) { t_up =  model->inter_data[i]->time; return (t_lo*D1/Mlo*D2/Nlo + t_up*D1/Mup*D2/Nlo)/2; }
			else if ( Mlo == model->inter_data[i]->M && Nup == model->inter_data[i]->N) { t_up =  model->inter_data[i]->time; return (t_lo*D1/Mlo*D2/Nlo + t_up*D1/Mlo*D2/Nup)/2; }
		}
		if (t_lo) return t_lo*D1/Mlo*D2/Nlo;
		error("GPUexec2Model_predict: Failed on simple interpolation...");
	}
	else error("GPUexec2Model_predict: Uninialized model");
}

/*
/// TODO: Direct Interpolation approach
GPUexec3Model_p GPUexec3Model_init_inter(short dev_id, char* func)
{
	GPUexec3Model_p out_model = (GPUexec3Model_p) malloc(sizeof(struct  BLAS3_data));
	inter_p inter_data[sample_lim_blas3];

	char filename[256];
	sprintf(filename, "%s/Data_manipulation/Results/%s/microbenchmarks/%s_gpu_microbench_%d_max-%d-%d-%d_step-%d-%d-%d_lim-%d.log", PROJECTDIR, MACHINE, func, dev_id, Mmax_blas3, Nmax_blas3, Kmax_blas3, M_step_blas3, N_step_blas3, K_step_blas3, sample_lim_blas3);
	FILE* fp = fopen(filename,"r");
	if (!fp) {
		fprintf(stderr, "GPUexec3Model_init: Logfile = %s\n", filename);
		error("GPUexec3Model_init: t_exec3 LogFile not generated");
	}
	size_t bench_lines = fmin( sample_lim_blas3, ((Mmax_blas3 - 512)/M_step_blas3 + 1)*((Kmax_blas3 - 1024)/K_step_blas3 + 1));
	fprintf(stderr, "GPUexec3Model_init: Reading %zu lines from %s\n", bench_lines, filename);
	double trashdata;
	int items;
	for (int i = 0; i < bench_lines; i++){   
		inter_data[i] = (inter_p) malloc(sizeof(struct interpolation_val));
		items = fscanf(fp, "%zu,%zu,%zu, %lf,%lf,%lf\n", &inter_data[i]->M, &inter_data[i]->N, &inter_data[i]->K, &inter_data[i]->time, &trashdata, &trashdata);
		//if (items != 7) error("GPUexec3Model_init_inter: Problem in reading model");
		//fprintf(stderr, "GPUexec3Model_init: Scanned entry %d: {M,N,K} = {%d,%d,%d} t = %lf ms\n", i, inter_data[i]->M, inter_data[i]->N, inter_data[i]->K, inter_data[i]->time*1000);
    	}
	out_model->inter_data = inter_data;
	out_model->inter_data_num =  bench_lines;
	out_model->pred_flag = 2; 
	out_model->dev_id = dev_id; 
	out_model->machine = MACHINE; 
	out_model->func = func; 

	return out_model;
}
*/

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
