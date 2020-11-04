///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief The communication sub-models used in CoCopeLia. 
///

#include <stdlib.h>
#include <math.h>

#include "cpu_utils.hpp"
#include "CoCopeLia_CoModel.hpp"

CoModel_p CoModel_init(short to, short from)
{
	CoModel_p out_model = (CoModel_p) malloc(sizeof(struct  comm_data));
	char filename[256];
	sprintf(filename, "%s/Data_manipulation/Results/%s/Models/transfer_model_%d_%d.log", PROJECTDIR, MACHINE, to, from);
	FILE* fp = fopen(filename,"r");
	if (!fp) {
		fprintf(stderr, "CoModel_init: Logfile = %s\n", filename);
		error("CoModel_init: t_comm LogFile not generated");
	}
	int items = fscanf(fp, "%Lf\n%Lf\n%Lf\n", &out_model->ti, &out_model->tb, &out_model->sl);
	if (items != 3) error("CoModel_init: Problem in reading model");
	fclose(fp);
	out_model->to = to; 
	out_model->from = from; 
	out_model->machine = MACHINE; 
	//fprintf(stderr, "CoModel_init : t_comm( %d -> %d) model initialized for %s -> ti =%e, tb=%e, sl = %e\n", out_model->from, out_model->to, out_model->machine, out_model->ti, out_model->tb, out_model->sl);
	return out_model;
}


/// Predict t_com for bytes using a Cmodel 
double t_com_predict(CoModel_p model, long double bytes)
{
	if (bytes <= 0) return 0;
	return model->ti + model-> tb*bytes; 
}

/// Predict t_com for bytes including bidirectional use slowdown
double t_com_sl(CoModel_p model, long double bytes)
{
	if (bytes <= 0) return 0;
	return model->ti + model->tb*bytes*model->sl; 
}


/// Predict t_com_bid for oposing transfers of bytes1,bytes2 
double t_com_bid_predict(CoModel_p model1, CoModel_p model2, long double bytes1, long double bytes2)
{
	//return fmax(t_com_predict(model1, bytes1), t_com_predict(model2, bytes2));
	if (bytes1 <= 0) return t_com_predict(model2, bytes2);
	else if (bytes2 <= 0) return t_com_predict(model1, bytes1);
	double t_sl1 = t_com_sl(model1,bytes1), t_sl2 = t_com_sl(model2,bytes2);
	if (t_sl1 < t_sl2) return t_sl1*( 1.0 - 1/model2->sl) + bytes2 * model2->tb + model2->ti/model2->sl; 
	else return t_sl2*( 1.0 - 1/model1->sl) + bytes1 * model1->tb + model1->ti/model1->sl; 
}

ComModel_p ComModel_init(short to, short from, short mode)
{
	ComModel_p out_model = (ComModel_p) malloc(sizeof(struct  comodel));
	char filename[256];
	sprintf(filename, "%s/BenchOutputs/%s/cublasSet_Get_to-%d_from-%d_min-%d_max-%d_step-%d_iter-%d.log", PROJECTDIR, MACHINE, to, from, minDim_trans, maxDim_trans, step_trans, ITER);
	FILE* fp = fopen(filename,"r");
	if (!fp) {
		fprintf(stderr, "CoModel_init: Logfile = %s\n", filename);
		error("CoModel_init: t_comm LogFile not generated");
	}
	size_t bench_lines = (maxDim_trans - minDim_trans)/step_trans + 1;
	fprintf(stderr, "CoModel_init: Reading %zu lines from %s\n", bench_lines, filename);
	int items;
	size_t trashdata, chech_tile = minDim_trans; 
	for (int i = 0; i < bench_lines; i++){
		items = fscanf(fp, "%zu, %lf,%lf,%lf, %lf,%lf,%lf\n", &trashdata, &out_model->av_time_buffer_tile[i], &out_model->min_time_buffer_tile[i], &out_model->max_time_buffer_tile[i], &out_model->av_time_sl_buffer_tile[i], &out_model->min_time_sl_buffer_tile[i], &out_model->max_time_sl_buffer_tile[i]);
		if (items != 7) error("CoModel_init: Problem in reading model");
		//fprintf(stderr, "GPUexec3Model_init: Scanned entry %d: T = %zu -> t_av = %lf ms, t_min = %lf ms, t_max = %lf ms\n", i, chech_tile, out_model->av_time_buffer_tile[i]*1000, out_model->min_time_buffer_tile[i]*1000, out_model->max_time_buffer_tile[i]*1000);
		chech_tile+=step_trans;
    	}

	out_model->mode = mode; 
	out_model->to = to; 
	out_model->from = from; 

	return out_model;
}

double CoTile_predict(ComModel_p model, size_t T, short dtype_sz)
{
	double result = 0, divider = 0; 
	// TODO: Only works for double, float for now. 
	if(dtype_sz == 8) divider = 1; 
	else if(dtype_sz == 4) divider = 2; 
	else error("CoTile_predict: Invalid datatype size"); 

	if( T<= 0 ) return result;  
	//if( T < 1024 ) warning("CoTile_predict: small dim in prediction");
	if (T < minDim_trans) error("CoTile_predict: Tile (T) smaller than micro-benchmark min");
	else if (T > maxDim_trans) error("CoTile_predict: Tile (T) larger than micro-benchmark max");
	else if ( (T - minDim_trans)%step_trans != 0) error("t_com_predict: Tile (T) not part of micro-benchmark");
	size_t offset = (T - minDim_trans)/step_trans;
	if (model->mode == 1) return model->av_time_buffer_tile[offset]/divider;
	else if (model->mode == 2) return model->min_time_buffer_tile[offset]/divider;
	else if (model->mode == 3) return model->max_time_buffer_tile[offset]/divider;
	else error("CoTile_predict: Uninialized model");
}

double CoTile_sl_predict(ComModel_p model, size_t T, short dtype_sz)
{
	double result = 0; 
	if( T<= 0 ) return result;  
	//if( T < 1024 ) warning("CoTile_sl_predict: small dim in prediction");
	if (T < minDim_trans) error("CoTile_sl_predict: Tile (T) smaller than micro-benchmark min");
	else if (T > maxDim_trans) error("CoTile_sl_predict: Tile (T) larger than micro-benchmark max");
	else if ( (T - minDim_trans)%step_trans != 0) error("t_com_sl: Tile (T) not part of micro-benchmark");
	size_t offset = (T - minDim_trans)/step_trans;
	if (model->mode == 1) return model->av_time_sl_buffer_tile[offset];
	else if (model->mode == 2) return model->min_time_sl_buffer_tile[offset];
	else if (model->mode == 3) return model->max_time_sl_buffer_tile[offset];
	else error("CoTile_sl_predict: Uninialized model");
}

double CoTile_bid_predict(ComModel_p model_h2d, ComModel_p model_d2h, size_t T, short dtype_sz, short numTin, short numTout)
{
	double t_acov_T, t_rem_T, sl_h2d, sl_d2h, sl_long;
	sl_h2d = CoTile_sl_predict(model_h2d, T, dtype_sz)/CoTile_predict(model_h2d, T, dtype_sz); 
	sl_d2h = CoTile_sl_predict(model_d2h, T, dtype_sz)/CoTile_predict(model_d2h, T, dtype_sz); 

	t_acov_T = fmin(numTin*CoTile_sl_predict(model_h2d, T, dtype_sz), numTout*CoTile_sl_predict(model_d2h, T, dtype_sz));

	if (numTin*CoTile_sl_predict(model_h2d, T, dtype_sz) > numTout*CoTile_sl_predict(model_d2h, T, dtype_sz)) sl_long = sl_h2d; 
	else sl_long = sl_d2h; 

	t_rem_T = (fmax(numTin*CoTile_sl_predict(model_h2d, T, dtype_sz), numTout*CoTile_sl_predict(model_d2h, T, dtype_sz)) - t_acov_T)/ sl_long; 
	return t_acov_T + t_rem_T; 
}

double CoVec_predict(ComModel_p model, size_t T, short dtype_sz)
{
	error("CoVec_predict: Not implemented");
	double result = 0, divider = 0; 
	// TODO: Only works for double, float for now. 
	if(dtype_sz == 8) divider = 1; 
	else if(dtype_sz == 4) divider = 2; 
	else error("CoVec_predict: Invalid datatype size"); 

	if( T<= 0 ) return result;  
	//if( T < 1024 ) warning("CoTile_predict: small dim in prediction");
	if (T < minDim_trans) error("CoTile_predict: Tile (T) smaller than micro-benchmark min");
	else if (T > maxDim_trans) error("CoTile_predict: Tile (T) larger than micro-benchmark max");
	else if ( (T - minDim_trans)%step_trans != 0) error("t_com_predict: Tile (T) not part of micro-benchmark");
	size_t offset = (T - minDim_trans)/step_trans;
	if (model->mode == 1) return model->av_time_buffer_vec[offset]/divider;
	else if (model->mode == 2) return model->min_time_buffer_vec[offset]/divider;
	else if (model->mode == 3) return model->max_time_buffer_vec[offset]/divider;
	else error("CoTile_predict: Uninialized model");
}

