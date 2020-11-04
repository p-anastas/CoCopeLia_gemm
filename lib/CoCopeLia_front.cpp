///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief A function which logs a model prediction for validation reasons. 
///
#include <stdlib.h>

#include "CoCopeLia_Model.hpp"
#include "Werkhoven.hpp"
#include "cpu_utils.hpp"

double log_CoCopelia_prediction(size_t D1, size_t D2, size_t D3, size_t T, short first_loc, short second_loc, short third_loc, short dev_id, char* func, short mode){
	double timer = csecond();

	char filename[256];
	if (mode == 1) sprintf(filename, "%s/Data_manipulation/Results/%s/validation/%s_CoCopelia_predict_avg_%d_v%s.log", PROJECTDIR, MACHINE, func, dev_id, VERSION);
	else if (mode == 2) sprintf(filename, "%s/Data_manipulation/Results/%s/validation/%s_CoCopelia_predict_min_%d_v%s.log", PROJECTDIR, MACHINE, func, dev_id, VERSION);
	else if (mode == 3) sprintf(filename, "%s/Data_manipulation/Results/%s/validation/%s_CoCopelia_predict_max_%d_v%s.log", PROJECTDIR, MACHINE, func, dev_id, VERSION);
	else if (mode == 4) sprintf(filename, "%s/Data_manipulation/Results/%s/validation/%s_CoCopelia_predict_no_reuse_%d_v%s.log", PROJECTDIR, MACHINE, func, dev_id, VERSION);

	CoCoModel_p CoComodel = NULL;

	if (!strcmp(func, "Dgemm")) CoComodel = CoCoModel_gemm_init(D1, D2, D3, first_loc, second_loc, third_loc, dev_id, func, mode%4 + mode/4);
	else if (!strcmp(func, "Sgemm")) CoComodel = CoCoModel_gemm_init(D1, D2, D3, first_loc, second_loc, third_loc, dev_id, func, mode%4 + mode/4);
	else if (!strcmp(func, "Dgemv")) error("log_CoCopelia_prediction: CoCoModel_gemv_init not implemented");//CoComodel = CoCoModel_gemv_init(D1, D2, first_loc, second_loc, third_loc, dev_id, func, mode);
	else if (!strcmp(func, "Daxpy")) error("log_CoCopelia_prediction: CoCoModel_axpy_init not implemented");//CoComodel = CoCoModel_axpy_init(D1, first_loc, second_loc, dev_id, func, mode);
	else error("log_CoCopelia_prediction: Invalid/Not implemented func");

	double coco;
	if (!strcmp(func, "Dgemm") || !strcmp(func, "Sgemm")){
		if (mode == 1 || mode == 2 || mode == 3) coco = CoCoModel_predict3(CoComodel, T);
		else if (mode == 4) coco = CoCoModel_noreuse_predict3(CoComodel, T);
		else error("log_CoCopelia_prediction: invalid mode");
	}
	else if (!strcmp(func, "Dgemv")){
		if (mode == 1 || mode == 2 || mode == 3) error("log_CoCopelia_prediction: CoCoModel_predict2 not implemented"); // coco = CoCoModel_predict2(CoComodel, T); 
		//else if (mode == 4) coco = CoCoModel_noreuse_predict2(CoComodel, DT1); // TODO: technically there is a no-reuse model for BLAS 2, its just insignificant to implement it 
		else error("log_CoCopelia_prediction: invalid mode");
	}
	else if (!strcmp(func, "Daxpy")){
		if (mode == 1 || mode == 2 || mode == 3) error("log_CoCopelia_prediction: CoCoModel_predict1 not implemented"); //coco = CoCoModel_predict1(CoComodel, T);
		else error("log_CoCopelia_prediction: invalid mode");
	}

	timer = csecond() - timer;

	fprintf(stderr, "Prediction time: %lf ms\n", timer);	

	WerkhovenModel_p werkmodel = NULL;
	double werp = 0; 
	if (!strcmp(func, "Dgemm")) {
		werkmodel = WerkhovenModel_init(dev_id, func, 3, mode%4 + mode/4);
		werp = WerkhovenModel_predict(werkmodel, first_loc*D1*D3*sizeof(double) + second_loc*D3*D2*sizeof(double) + third_loc*D1*D2*sizeof(double), third_loc*D1*D2*sizeof(double), (1.0*D1/T)*(1.0*D2/T)*(1.0*D3/T), 3, D1,D2,D3);
	}
	else if (!strcmp(func, "Sgemm")){
		werkmodel = WerkhovenModel_init(dev_id, func, 3, mode%4 + mode/4);
		werp = WerkhovenModel_predict(werkmodel, first_loc*D1*D3*sizeof(float) + second_loc*D3*D2*sizeof(float) + third_loc*D1*D2*sizeof(float), third_loc*D1*D2*sizeof(float), (1.0*D1/T)*(1.0*D2/T)*(1.0*D3/T), 3, D1,D2,D3);
	}
	else if (!strcmp(func, "Dgemv")){
		werkmodel = WerkhovenModel_init(dev_id, func, 2, mode); 
		werp = WerkhovenModel_predict(werkmodel, first_loc*D1*D2*sizeof(double) + second_loc*D2*sizeof(double) + third_loc*D1*sizeof(double), third_loc*D1*sizeof(double), (1.0*D1/T), 2, D1,D2,-1);
	}
	else if (!strcmp(func, "Daxpy")){
		werkmodel = WerkhovenModel_init(dev_id, func, 1, mode);
		werp = WerkhovenModel_predict(werkmodel, first_loc*D1*sizeof(double) + second_loc*D1*sizeof(double), second_loc*D1*sizeof(double), (1.0*D1/T), 1, D1,-1,-1);
	}
	else error("log_CoCopelia_prediction: Invalid/Not implemented func");

	FILE* fp = fopen(filename,"a");
	if (!fp) error("report_results: LogFile failed to open");
	if (!strcmp(func, "Dgemm") || !strcmp(func, "Sgemm") ) fprintf(fp,"%zu,%zu,%zu,%zu,%d,%d,%d, %e,%e\n", D1, D2, D3, T, first_loc, second_loc, third_loc, werp, coco);
	else if (!strcmp(func, "Dgemv")) fprintf(fp,"%zu,%zu,%zu,%d,%d,%d, %e,%e\n", D1, D2, T, first_loc, second_loc, third_loc, werp, coco); 
	else if (!strcmp(func, "Daxpy")) fprintf(fp,"%zu,%zu,%d,%d, %e,%e\n", D1, T, first_loc, second_loc, werp, coco);
	else error("log_CoCopelia_prediction: Invalid/Not implemented func");
        fclose(fp); 
	free(CoComodel);
	free(werkmodel);

	return timer; 
}

