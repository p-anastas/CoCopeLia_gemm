///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief Some testing helper functions for file output.
///

#include <cstdlib>

#include "cpu_utils.hpp"

void check_benchmark(char *filename){
	FILE* fp = fopen(filename,"r");
	if (!fp) { 
		fp = fopen(filename,"w+");
		if (!fp) error("report_results: LogFile failed to open");
		else warning("Generating Logfile...");
		fclose(fp);
	}
	else {
		fprintf(stderr,"Benchmark@%s found: %s\n", MACHINE, filename);
		fclose(fp);	
		exit(1); 
	}
	return;		  	
}

void check_log(char* filename, size_t D1, size_t D2, size_t D3, int T, short loc1, short loc2, short loc3, short dev_id){
	FILE* fp = fopen(filename,"r");
	if (!fp) { 
		fp = fopen(filename,"w+");
		if (!fp) error("report_results: LogFile failed to open");
		else warning("Generating Logfile...");
	}	
	char buffer[256], search_string[256];
	sprintf(search_string, "%zu,%zu,%zu,%d,%d,%d,%d,%d", D1, D2, D3, T, loc1, loc2, loc3, dev_id);
	while (fgets(buffer, sizeof(buffer), fp) != NULL){
		if(strstr(buffer, search_string) != NULL){
   			fprintf(stderr,"%zu,%zu,%zu,%d,%d,%d,%d Entry found...quiting\n", D1, D2, D3, loc1, loc2, loc3, dev_id);
			fclose(fp);	
			exit(1); 
		}
	}			
    	fclose(fp);
	return;
}

void check_log_ser(char* filename, size_t D1, size_t D2, size_t D3, short loc1, short loc2, short loc3, short dev_id){
	FILE* fp = fopen(filename,"r");
	if (!fp) { 
		fp = fopen(filename,"w+");
		if (!fp) error("report_results: LogFile failed to open");
		else warning("Generating Logfile...");
	}	
	char buffer[256], search_string[256];
	sprintf(search_string, "%zu,%zu,%zu,%d,%d,%d,%d", D1, D2, D3, loc1, loc2, loc3, dev_id);
	while (fgets(buffer, sizeof(buffer), fp) != NULL){
		if(strstr(buffer, search_string) != NULL){
   			fprintf(stderr,"%zu,%zu,%zu,%d,%d,%d,%d Entry found...quiting\n", D1, D2, D3, loc1, loc2, loc3, dev_id);
			fclose(fp);	
			exit(1); 
		}
	}			
    	fclose(fp);
	return;
}

void report_results(char* filename, size_t D1, size_t D2, size_t D3, int T, short loc1, short loc2, short loc3, short dev_id, double av_time, double min_time, double max_time, double init_over_time){
	FILE* fp = fopen(filename,"a");
	if (!fp) error("report_results: LogFile failed to open");
   	fprintf(fp,"%zu,%zu,%zu,%d,%d,%d,%d,%d, %e,%e,%e,%e\n", D1, D2, D3, T, loc1, loc2, loc3, dev_id, av_time, min_time, max_time, init_over_time);
        fclose(fp); 
	return;
}

void report_results_ser(char* filename, size_t D1, size_t D2, size_t D3, short loc1, short loc2, short loc3, short dev_id, double av_ex_time, double min_ex_time, double max_ex_time, double init_over_ex_time, double send_time, double get_time){
	FILE* fp = fopen(filename,"a");
	if (!fp) error("report_results: LogFile failed to open");
	double transfer_t =  send_time + get_time;
   	fprintf(fp,"%zu,%zu,%zu,%d,%d,%d,%d, %e,%e,%e,%e,%e,%e\n", D1, D2, D3, loc1, loc2, loc3, dev_id, av_ex_time + transfer_t, min_ex_time + transfer_t, max_ex_time + transfer_t, init_over_ex_time + transfer_t, send_time, get_time);
        fclose(fp); 
	return;
}
