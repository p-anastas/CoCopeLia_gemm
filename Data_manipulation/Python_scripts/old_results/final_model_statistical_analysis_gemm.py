import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import math
import numpy as np

machine_names= ['testbed-II_Tesla-V100'] #'testbed-I_Tesla-K40', 
funcs=['Dgemm','Sgemm'] # 
version='1.2'
dataset_condis=[(validata['K']*validata['M']*validata['N'] <= (size**3)) & ((validata['K']+1)*(validata['M']+1)*(validata['N']+1) >= (size-1)**3)

for func in funcs:
	for machine in machine_names:

		validata_CoCo = pd.read_csv('../Results/%s/validation/CoCopeLia_%s_0_v%s.log' % (machine, func, version),
							   header =None, usecols = [0,1,2,3,4,5,6,8], names= ['M','N','K','T', 'Aloc','Bloc', 'Cloc',  'CoCopeLia_t'])
		validata_cuBLASXt = pd.read_csv('../Results/%s/validation/cuBLASXt_%s_0.log' % (machine, func),
							   header =None, usecols = [0,1,2,3,4,5,6,8], names= ['M','N','K','T', 'Aloc','Bloc', 'Cloc', 'cublasXt_t'])

		print( "Read %d values from \"../Results/%s/validation/CoCopeLia_%s_0_v%s.log\"..." %( len(validata_CoCo), machine, func, version))
		print( "Read %d values from \"../Results/%s/validation/cuBLASXt_%s_0.log\".." %( len(validata_cuBLASXt), machine, func))

		validata = pd.merge(validata_CoCo, validata_cuBLASXt, on = ['M', 'N', 'K', 'T', 'Aloc', 'Bloc', 'Cloc'])
		print(validata.head(1))
		#print(validata['CoCopeLia_t'].min())

		pred_data = pd.read_csv('../Results/%s/validation/%s_CoCopelia_predict_avg_0_v%s.log' % (machine, func, version),
								header =None,  names= ['M','N','K','T', 'Aloc','Bloc', 'Cloc', 'werkhoven', 'CoCopelia'], dtype={'werkhoven': np.float64, 'CoCopelia': np.float64} ) #usecols = [0,1,2,3,5,6,7,8,9],

		print( "Read %d predictions from \"../Results/%s/validation/%s_CoCopelia_predict_avg_0.log\".." %( len(pred_data), machine, func))

		#print(pred_data.head(1))
		#print(pred_data['CoCopelia'].max())

		merged_full = pd.merge(validata,pred_data, on = ['M', 'N', 'K', 'T', 'Aloc', 'Bloc', 'Cloc'])
		print( "Combined %d prediction/validation pairs" %( len(merged_full)))


		merged = merged_full[#((merged_full['K'] != 4096) | (merged_full['M'] == 32768)) &
					(merged_full['T'] > 512) &  
					#(merged_full['Aloc'] ==  1 ) & (merged_full['Bloc'] ==  1) & (merged_full['Cloc'] ==  1) & 
					(merged_full['M']/1.5 > merged_full['T']) & (merged_full['N']/1.5 > merged_full['T']) & (merged_full['K']/1.5 > merged_full['T'])]
		print( "Kept %d pairs in the final validation set" %( len(merged)))			
		#print(merged.head(1))
		merged['APE_CoCopeLia'] = 100*abs(merged['CoCopeLia_t'] - merged['CoCopelia'])/ merged['CoCopeLia_t']
		merged['APE_werkhoven'] = 100*abs(merged['CoCopeLia_t'] - merged['werkhoven'])/ merged['CoCopeLia_t']

		merged_clean = merged#[(merged['APE_CoCopeLia'] < 25) & (merged['APE_werkhoven'] < 25)]
		#merged_werk_outliers = merged[(merged['APE_werkhoven'] >= 25) ]
		#merged_CoCo_outliers = merged[(merged['APE_CoCopeLia'] >= 25) ]
		#print( "Kept %d pairs in after cleaning final validation set from outliers (0.25)" %( len(merged_clean)))
		#print( "From %d Werkhoven outliers:"	%( len(merged_werk_outliers)))
		#print(merged_werk_outliers)
		#print( "From %d CoCopeLia outliers:"	%( len(merged_CoCo_outliers)))
		#print(merged_CoCo_outliers)
		#print( "CoCopelia MAPE without outliers (0.25): %lf" % merged_clean['APE_CoCopeLia'].mean())
		#print( "Werkhoven MAPE without outliers (0.25): %lf" % merged_clean['APE_werkhoven'].mean())

		print('Function: %s Machine : %s' % (func, machine))
		print( "CoCopelia MAPE : %lf" % merged['APE_CoCopeLia'].mean())
		print( "Werkhoven MAPE : %lf" % merged['APE_werkhoven'].mean())
		print("\n")
		teams = merged.groupby(['M', 'N' , 'K', 'Aloc', 'Bloc', 'Cloc'])
