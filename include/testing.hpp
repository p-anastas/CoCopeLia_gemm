///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief The start of Zawarudo
///

#include <stdio.h>
#include <cstring>

void check_benchmark(char *filename);

void check_log(char* filename, size_t D1, size_t D2, size_t D3, int T, short loc1, short loc2, short loc3, short dev_id);

void check_log_ser(char* filename, size_t D1, size_t D2, size_t D3, short loc1, short loc2, short loc3, short dev_id);

void report_results(char* filename, size_t D1, size_t D2, size_t D3, int T, short loc1, short loc2, short loc3, short dev_id, double av_time, double min_time, double max_time, double init_over_time);

void report_results_ser(char* filename, size_t D1, size_t D2, size_t D3, short loc1, short loc2, short loc3, short dev_id, double av_ex_time, double min_ex_time, double max_ex_time, double init_over_ex_time, double send_time, double get_time);
