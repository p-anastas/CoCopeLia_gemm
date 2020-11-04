import pandas as pd
import math

machine_names= ['testbed-II_Tesla-V100']
funcs=['Dgemm']
version='1.1'

for func in funcs:
	for machine in machine_names:

		validata_CoCo = pd.read_csv('../Results/%s/validation/CoCopeLia_%s_0_v%s.log' % (machine, func, version),
							   header =None, usecols = [0,1,2,3,4,5,6,8], names= ['M','N','K','T', 'Aloc','Bloc', 'Cloc',  'CoCopeLia_t'])
		validata_cuBLASXt = pd.read_csv('../Results/%s/validation/cuBLASXt_%s_0.log' % (machine, func),
							   header =None, usecols = [0,1,2,3,4,5,6,8], names= ['M','N','K','T', 'Aloc','Bloc', 'Cloc', 'cublasXt_t'])

		validata = pd.merge(validata_CoCo, validata_cuBLASXt, on = ['M', 'N', 'K', 'T', 'Aloc', 'Bloc', 'Cloc'])
		print(validata.head(1))
		print(validata['CoCopeLia_t'].min())

		pred_data = pd.read_csv('../Results/%s/validation/%s_CoCopelia_predict_avg_0.log' % (machine, func),
								header =None,  names= ['M','N','K','T', 'Aloc','Bloc', 'Cloc', 'werkhoven', 'CoCopelia']) #usecols = [0,1,2,3,5,6,7,8,9],
		#print(pred_data.head(1))
		#print(pred_data['CoCopelia'].max())

		merged_full = pd.merge(validata,pred_data, on = ['M', 'N', 'K', 'T', 'Aloc', 'Bloc', 'Cloc'])
		validata['size'] = validata['M']*validata['N']*validata['K']
		merged = merged_full[((validata['K'] != 4096) | (validata['M'] == 32768)) &
					(validata['size'] < validata['T']**3)]
		merged['APE_CoCopeLia'] = 100*abs(merged['CoCopeLia_t'] - merged['CoCopelia'])/ merged['CoCopeLia_t']
		merged['APE_werkhoven'] = 100*abs(merged['CoCopeLia_t'] - merged['werkhoven'])/ merged['CoCopeLia_t']
		#merged['PE_CoCoBLAS'] = 100*(merged['streamed_t'] - merged['CoCopelia'])/ merged['streamed_t']
		#merged['PE_werkhoven'] = 100*(merged['streamed_t'] - merged['werkhoven'])/ merged['streamed_t']
		#print(merged.iloc[merged['CoCopelia'].argmax()])
		#merged_under_co = merged[merged['PE_CoCoBLAS'] > 0]
		#merged_over_co = merged[merged['PE_CoCoBLAS'] <= 0]
		#merged_under_we = merged[merged['PE_werkhoven'] > 0]
		#merged_over_we = merged[merged['PE_werkhoven'] <= 0]
		print('Function: %s Machine : %s' % (func, machine))
		#print( "CoCopelia MAPE : %lf, NMPE : %lf, PMPE : %lf" % (merged['APE_CoCoBLAS'].mean(),merged_under_co['PE_CoCoBLAS'].mean(),merged_over_co['PE_CoCoBLAS'].mean()))
		#print( "Werkhoven MAPE : %lf, NMPE : %lf, PMPE : %lf" % (merged['APE_werkhoven'].mean(),merged_under_we['PE_werkhoven'].mean(),merged_over_we['PE_werkhoven'].mean()))
		print( "CoCopelia MAPE : %lf" % merged['APE_CoCopeLia'].mean())
		print( "Werkhoven MAPE : %lf" % merged['APE_werkhoven'].mean())
		teams = merged.groupby(['M', 'N' , 'K', 'Aloc', 'Bloc', 'Cloc'])
