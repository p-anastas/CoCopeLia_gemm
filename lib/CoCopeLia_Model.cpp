///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief The 3-way concurency overlap prediction models for BLAS
///

#include <stdlib.h>
#include <math.h>

#include "CoCopeLia_CoModel.hpp"
#include "CoCopeLia_GPUexec.hpp"
#include "CoCopeLia_Model.hpp"
#include "cpu_utils.hpp"

///  A naive estimator which chooses if output (d2h) should be overlapped with input (h2d) or performed afterwards (0=yes,1=no)
short CoCoModel_choose_transfer_mode3(CoCoModel_p model, size_t T){
	short mode = 0; 
	double t_exec_total = (1.0*model->D1/T)*(1.0*model->D2/T)*(1.0*model->D3/T)*GPUexec3Model_predict((GPUexec3Model_p) model->GPUexec_model_ptr, T);
	double t_in_total = (1.0*model->D1/T*model->D2/T*model->V->in[2] * model->V->loc[2] + 1.0*model->D1/T*model->D3/T*model->V->in[0] * model->V->loc[0] + 1.0*model->D2/T*model->D3/T*model->V->in[1] * model->V->loc[1])*t_com_predict(model->h2d, T*T*model->V->dtype_sz); //CoTile_predict(model->h2d, T, model->V->dtype_sz);	
	double t_out_total = 1.0*model->D1/T*model->D2/T*model->V->out[2] * model->V->out_loc[2] *  t_com_predict(model->d2h, T*T*model->V->dtype_sz);//CoTile_predict(model->d2h, T, model->V->dtype_sz);

	if (t_exec_total < (t_in_total + t_out_total)) mode = 0; 
	else mode = 1;

	fprintf(stderr, "Selected mode %d for h2d/d2h overlap\n", mode);
	fprintf(stderr, "Based on t_exec=%lf, t_in=%lf + t_out=%lf \n", t_exec_total, t_in_total, t_out_total);

	return mode; 
}

///  Predicts 3-way overlaped execution time for BLAS3 Square tilling blocking.
double CoCoModel_predict3(CoCoModel_p model, size_t T)
{
	//fprintf(stderr, "\nCoCoModel_predict3 ->\nProblem dims: D1 = %zu, D2 = %zu, D3 = %zu\nVdata(%d)->\n", model->D1, model->D2, model->D3, model->V->numT);

	double t_h2d_T3 = 0, t_d2h_T3 = 0, t_exec_T3 = 0, t_total = 0;
	t_exec_T3 = GPUexec3Model_predict((GPUexec3Model_p) model->GPUexec_model_ptr, T);
	t_h2d_T3 = t_com_predict(model->h2d, T*T*model->V->dtype_sz); //CoTile_predict(model->h2d, T, model->V->dtype_sz);	
	t_d2h_T3 = t_com_predict(model->d2h, T*T*model->V->dtype_sz);//CoTile_predict(model->d2h, T, model->V->dtype_sz);

	double t_in_T, t_out_T;
	size_t numTin = 0, numTout = 0;
	
	double zero_over = 0, one_over = 0, two_over = 0; 
	zero_over =  (1.0*model->D1/T)*(1.0*model->D2/T)*(1.0*model->D3/T) - 1;
	for (int i = 0; i < model->V->numT; i++){
		if (*model->V->Dim1[i] < 1 || *model->V->Dim2[i] < 1) error("CoCoModel_predict3: Invalid data struct dims");
		numTin += model->V->in[i] * model->V->loc[i]; 
		numTout += model->V->out[i] * model->V->loc[i]; 
		one_over+= model->V->in[i] * model->V->loc[i]*((1.0*(*model->V->Dim1[i]))/T)*((1.0*(*model->V->Dim2[i]))/T - 1); // The -1 only if two_over is ommited
				
		if (t_h2d_T3 > t_exec_T3) {
		// two_over kernels calculated
			for (int j = i + 1; j < model->V->numT; j++)
				if (model->V->in[i] * model->V->loc[i] && model->V->in[j] * model->V->loc[j]){
					if ( model->V->Dim1[i] == model->V->Dim1[j] || model->V->Dim1[i] == model->V->Dim2[j]) two_over += ((1.0*(*model->V->Dim1[i]))/T) - 1; 
					else if ( model->V->Dim2[i] == model->V->Dim1[j] || model->V->Dim2[i] == model->V->Dim2[j]) two_over += ((1.0*(*model->V->Dim2[i]))/T) - 1; 
					else error("CoCoModel_predict3: something is wrong with my brilliant pointer comparisson idea");
			}
		} 
	}	
	// Performance Cheat
	if ( 2* t_h2d_T3 > t_exec_T3 && t_exec_T3 > t_h2d_T3)  two_over += (1.0*model->D3/T); 
	one_over -= (2*two_over + numTin); 
	zero_over -= (one_over + two_over); 
	t_total = t_exec_T3*(1 + zero_over) + 
	fmax(t_exec_T3, t_h2d_T3)* one_over +
	fmax(t_exec_T3, t_h2d_T3*2)* two_over +
	+ numTin * t_h2d_T3 + numTout * t_d2h_T3;

	fprintf(stderr, "CoCopelia (T=%d) predicted :\n"
	"\tt_h2d_T3: %lf ms ( %lf Gb/s)\n"
	"\t -> two_over = %d -> one_over = %d -> zero_over = %d\n"
	"\tt_execT3: %lf ms (%lf GFlops/s)\n"
	"\tt_d2h_T3: %lf ms ( %lf Gb/s)\n"
	"\tt_total: %lf ms (%lf GFlops/s)\n\n", 
	T, t_h2d_T3*1000, Gval_per_s(T*T*model->V->dtype_sz,t_h2d_T3),  
	two_over, one_over, zero_over,
	t_exec_T3*1000, Gval_per_s(dgemm_flops(T,T,T), t_exec_T3), 
	t_d2h_T3*1000, Gval_per_s(T*T*model->V->dtype_sz,t_d2h_T3),
	t_total*1000, Gval_per_s(dgemm_flops(model->D1,model->D2,model->D3), t_total));
		
	return t_total; 
}

///  Predicts 3-way overlaped execution time for BLAS3 Square tilling blocking without data reuse.
double CoCoModel_noreuse_predict3(CoCoModel_p model, size_t T)
{
	double t_h2d_T3 = 0, t_d2h_T3 = 0, t_exec_T3 = 0, t_total = 0, t_over_T3 = 0;
	t_exec_T3 = GPUexec3Model_predict((GPUexec3Model_p) model->GPUexec_model_ptr, T);
	t_h2d_T3 = t_com_predict(model->h2d, T*T*model->V->dtype_sz); //CoTile_predict(model->h2d, T, model->V->dtype_sz);	
	t_d2h_T3 = t_com_predict(model->d2h, T*T*model->V->dtype_sz);//CoTile_predict(model->d2h, T, model->V->dtype_sz);

	double t_in_T, t_out_T;
	size_t numTin = 0, numTout = 0;
	
	double ker_over =  (1.0*model->D1/T)*(1.0*model->D2/T)*(1.0*model->D3/T) - 1;
	for (int i = 0; i < model->V->numT; i++){
		if (*model->V->Dim1[i] < 1 || *model->V->Dim2[i] < 1) error("CoCoModel_predict3: Invalid data struct dims");
		numTin += model->V->in[i] * model->V->loc[i]; 
		numTout += model->V->out[i] * model->V->loc[i];
	}
	// Use bidirectional magic here if needed
	t_over_T3 = fmax(numTin*t_h2d_T3, t_d2h_T3*numTout);//CoTile_bid_predict(model->h2d, model->d2h, T, model->V->dtype_sz, numTin, numTout);
	t_total = fmax(t_exec_T3, t_over_T3)* ker_over +
	+ t_exec_T3 + numTin * t_h2d_T3 + numTout * t_d2h_T3;
	return t_total;
}

///  Initializes the model for gemm
CoCoModel_p CoCoModel_gemm_init(size_t M, size_t N, size_t K, short A_loc, short B_loc, short C_loc, short dev_id, char* func, short mode){
	CoCoModel_p out_model = (CoCoModel_p) malloc(sizeof(struct CoCo_model));
	out_model->h2d = CoModel_init(dev_id, -1);//ComModel_init(dev_id, -1, mode);
	out_model->d2h = CoModel_init(-1, dev_id);//ComModel_init(-1, dev_id, mode);
	out_model->GPUexec_model_ptr = (void*) GPUexec3Model_init(dev_id, func, mode);
	out_model->V = (Vdata_p) malloc(sizeof(struct V_struct));

	// Gemm Routine info
	out_model->V->numT = 3;

	if (!strcmp(func, "Dgemm")) out_model->V->dtype_sz = sizeof(double);
	else if (!strcmp(func, "Sgemm")) out_model->V->dtype_sz = sizeof(float);

	out_model->V->in[0] = 1; 
	out_model->V->in[1] = 1; 
	out_model->V->in[2] = 1; 

	out_model->V->out[0] = 0;
	out_model->V->out[1] = 0;
	out_model->V->out[2] = 1;

	// Gemm Problem Specific values for Routine info functions
	out_model->D1 = M;
	out_model->D2 = N;
	out_model->D3 = K;

	out_model->V->Dim1[0] = &out_model->D1;
	out_model->V->Dim1[1] = &out_model->D3;
	out_model->V->Dim1[2] = &out_model->D1;

	out_model->V->Dim2[0] = &out_model->D3;
	out_model->V->Dim2[1] = &out_model->D2;
	out_model->V->Dim2[2] = &out_model->D2;

	out_model->V->loc[0] = A_loc;
	out_model->V->loc[1] = B_loc;
	out_model->V->loc[2] = C_loc;

	out_model->V->out_loc[0] = A_loc;
	out_model->V->out_loc[1] = B_loc;
	out_model->V->out_loc[2] = C_loc;

	fprintf(stderr, "CoCoModel_gemm initalized for %s->\nInitial problem dims: D1 = %zu, D2 = %zu, D3 = %zu\n"
	"Data tiles : A(%zu,%zu), B(%zu,%zu), C(%zu,%zu) in loc (%d,%d,%d)\n", \
	func, out_model->D1, out_model->D2, out_model->D3, out_model->D1, out_model->D3, out_model->D3, out_model->D2, out_model->D1, out_model->D2, out_model->V->out_loc[0], out_model->V->out_loc[1], out_model->V->out_loc[2]);

	return out_model;
}

///  Itterates through benchmarked values for T and chooses the Tbest that minimizes total time. 
size_t CoCoModel_optimize3(CoCoModel_p model){
	size_t min_T = minDim_blas3, max_allowed_T = fmin(fmin(model->D1, model->D2),model->D3);
	if (min_T > max_allowed_T) return max_allowed_T;
	double temp_t, min_t = CoCoModel_predict3(model, min_T);
	for (size_t trial_T = min_T + step_blas3; trial_T < (size_t) fmin(max_allowed_T,maxDim_blas3) + 1; trial_T += step_blas3){
		temp_t = CoCoModel_predict3(model, trial_T);
		//fprintf(stderr, "Checking T = %zu\n : t = %lf ms\n", trial_T, temp_t*1000);	
		if ( temp_t < min_t ){
			min_t = temp_t; 
			min_T = trial_T;
		}
	}
	fprintf(stderr, "CoCoModel_optimize3 T = %zu\n : t_min = %lf ms\n", min_T, min_t*1000);	
	return min_T; 
}

