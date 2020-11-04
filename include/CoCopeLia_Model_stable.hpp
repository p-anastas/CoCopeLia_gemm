///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief Some CUDA function calls with added error-checking
///
#ifndef COCOPELIA_MODEL_H
#define COCOPELIA_MODEL_H

#include "CoCopeLia_CoModel.hpp"

// TODO: To avoid mallocs, define a set vec size of 4 (No BLAS has that many data arguments anyway)
typedef struct V_struct{	
	// Routine specific
	short numT;
	short dtype_sz;
	short in[4]; // Possibly modified from scalar_nz
	short out[4]; // Possibly modified from scalar_nz

	// Problem specific
	size_t Dim1[4];
	size_t Dim2[4];
	short loc[4];
	short out_loc[4];

}* Vdata_p;

typedef struct CoCo_model{
	Vdata_p V;
	ComModel_p h2d;
	ComModel_p d2h;
	double Ker_pot; 
	size_t D1 = 1, D2 = 1, D3 = 1; 
	void* GPUexec_model_ptr;
	// FIXME: Add cpu_exec prediction

}* CoCoModel_p;

///  Predicts 3-way overlaped execution time for BLAS1 1-dim blocking.
double CoCoModel_predict1(CoCoModel_p model, long long DT1);

///  Predicts 3-way overlaped execution time for BLAS2 1-dim blocking.
double CoCoModel_predict2(CoCoModel_p model, long long DT1);

///  Predicts 3-way overlaped execution time for BLAS3 2-dim blocking.
double CoCoModel_predict3(CoCoModel_p model, size_t T);

///  Predicts 3-way overlaped execution time for BLAS3 2-dim blocking without data reuse.
double CoCoModel_noreuse_predict3(CoCoModel_p model, size_t T);

///  Predicts Best tile size for 3-way overlaped execution time for BLAS3 2-dim blocking.
size_t CoCoModel_optimize3(CoCoModel_p model);

/// Choose the best way to approach h2d/d2h overlap 
short CoCoModel_choose_transfer_mode3(CoCoModel_p model, size_t T); 

///  Predicts Best tile size for 3-way overlaped execution time for BLAS1 1-dim blocking.
size_t CoCoModel_optimize1(CoCoModel_p model);

CoCoModel_p CoCoModel_gemm_init(size_t M, size_t N, size_t K, short A_loc, short B_loc, short C_loc, short dev_id, char* func, short mode);

CoCoModel_p CoCoModel_gemv_init(size_t M, size_t N, short A_loc, short x_loc, short y_loc, short dev_id, char* func, short mode);

CoCoModel_p CoCoModel_axpy_init(size_t N, short x_loc, short y_loc, short dev_id, char* func, short mode);

#endif
