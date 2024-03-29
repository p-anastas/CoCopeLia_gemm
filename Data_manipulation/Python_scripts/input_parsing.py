import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
from scipy import stats

def get_level(function):
	if function == 'Daxpy':
		level = 1
	elif ((function == 'Dgemm') or (function == 'Sgemm')):
		level = 3
	else:
		print("get_level: Error (Invalid function)")
		quit()
	return level

def read_validation_values(rootdir,machine,func,version):
	level = get_level(func)
	if level==3:
		validata_CoCo = pd.read_csv('%s/Results/%s/validation/CoCopeLia_%s_0_v%s.log' % (rootdir, machine, func, version),
				header =None, usecols = [0,1,2,3,4,5,6,8], names= ['M','N','K','T', 'Aloc','Bloc', 'Cloc',  'CoCopeLia_t'])
		#We use cublasXt_t as a no-reuse example
		validata_cuBLASXt = pd.read_csv('%s/Results/%s/validation/cuBLASXt_%s_0.log' % (rootdir, machine, func),
				header =None, usecols = [0,1,2,3,4,5,6,8], names= ['M','N','K','T', 'Aloc','Bloc', 'Cloc', 'Noreuse_t'])

		print( "read_validation_values : Read %d values from \"%s/Results/%s/validation/CoCopeLia_%s_0_v%s.log\"..." %( len(validata_CoCo), rootdir,machine, func, version))
		print( "read_validation_values : Read %d values from \"%s/Results/%s/validation/cuBLASXt_%s_0.log\".." %( len(validata_cuBLASXt), rootdir,machine, func))

		validata = pd.merge(validata_CoCo, validata_cuBLASXt, on = ['M', 'N', 'K', 'T', 'Aloc', 'Bloc', 'Cloc'])

	elif level==1: 
		validata = pd.read_csv('%s/Results/%s/validation/CoCopeLia_%s_0_v%s.log' % (rootdir, machine, func, version),
							   header =None, usecols = [0,1,2,3,8,9], names= ['N', 'T', 'Aloc','Bloc', 'Noreuse_t', 'unified_t'])
		validata['M'] = validata['K'] = validata['Cloc'] = -1
		print( "read_validation_values : Read %d values from \"%s/Results/%s/validation/CoCopeLia_%s_0_v%s.log\"..." %( len(validata), rootdir,machine, func, version))
	return validata

def read_validation_predictions(rootdir,machine,func,version):
	level = get_level(func)
	if level==3:
		pred_data_reuse = pd.read_csv('%s/Results/%s/validation/%s_CoCopelia_predict_avg_0_v%s.log' % (rootdir,machine, func, version),
				header =None,  names= ['M','N','K','T', 'Aloc','Bloc', 'Cloc', 'werkhoven', 'CoCopelia'], dtype={'werkhoven': np.float64, 'CoCopelia': np.float64} ) #usecols = [0,1,2,3,5,6,7,8,9],

		pred_data_no_reuse = pd.read_csv('%s/Results/%s/validation/%s_CoCopelia_predict_no_reuse_0_v%s.log' % (rootdir,machine, func, version),
				header =None,  names= ['M','N','K','T', 'Aloc','Bloc', 'Cloc', 'werkhoven', 'CoCopelia_nr'], dtype={'werkhoven': np.float64, 'CoCopelia_nr': np.float64} ) #usecols = [0,1,2,3,5,6,7,8,9],

		pred_data = pd.merge(pred_data_reuse,pred_data_no_reuse, on = ['M', 'N', 'K', 'T', 'Aloc', 'Bloc', 'Cloc', 'werkhoven'])

	elif level==1: 
		pred_data = pd.read_csv('%s/Results/%s/validation/%s_CoCopelia_predict_avg_0_v%s.log' % (rootdir,machine, func, version),
				header =None,  names= ['N','T', 'Aloc','Bloc', 'werkhoven', 'CoCopelia_nr'], dtype={'werkhoven': np.float64, 'CoCopelia': np.float64} ) #usecols = [0,1,2,3,5,6,7,8,9],
		pred_data['M'] = pred_data['K'] = pred_data['Cloc'] = -1

	print( "read_validation_predictions : Read %d predictions from \"%s/Results/%s/validation/%s_CoCopelia_predict_avg_0_v%s.log\".." %( len(pred_data), rootdir, machine, func, version))	
	return pred_data

def create_validation_set(values_df, pred_df,func):
	level = get_level(func)
	merged_full = pd.merge(values_df,pred_df, on = ['M', 'N', 'K', 'T', 'Aloc', 'Bloc', 'Cloc'])

	print( "create_validation_set : Combined %d prediction/validation pairs" %( len(merged_full)))
	if level==3:
		merged = merged_full[(merged_full['T'] != 8192) & (merged_full['T'] >= 512) &  (merged_full['M']/1.5 >= merged_full['T']) & (merged_full['N']/1.5 >= merged_full['T']) & (merged_full['K']/1.5 >= merged_full['T'])] # Remove for perper
	elif level==1: 
		merged = merged_full[(merged_full['T'] >= merged_full['N']/64) &  (merged_full['T'] <= merged_full['N']/1.5)]
					#(merged_full['Aloc'] ==  1 ) & (merged_full['Bloc'] ==  1) & (merged_full['Cloc'] ==  1) & 	

	print( "create_validation_set: %d pairs kept in the combined validation set" %( len(merged)))
	return merged

def validation_set_split_BLAS3(name,locs,mid_sizes,ctrs_fat,ctrs_thin,input_set):
	cond_total = False
	for loc in locs:
		cond_loc = ((input_set['Aloc'] ==  loc[0] ) & (input_set['Bloc'] ==  loc[1]) & (input_set['Cloc'] ==  loc[2]))
		for mid_size in mid_sizes:
			cond_sz = ((input_set['K']*input_set['M']*input_set['N'] <= (mid_size**3)) & ((input_set['K']+1)*(input_set['M']+1)*(input_set['N']+1) >= (mid_size-1)**3))
			for ctr_fat in ctrs_fat:
				#set_row = input_set[(input_set['Aloc'] ==  loc[0] ) & (input_set['Bloc'] ==  loc[1]) & (input_set['Cloc'] ==  loc[2]) & 
							#(input_set['K']*input_set['M']*input_set['N'] <= (mid_size**3)) & ((input_set['K']+1)*(input_set['M']+1)*(input_set['N']+1) >= (mid_size-1)**3) &
				cond_fat = ((input_set['M'] == input_set['N']) & (input_set['N'] >= input_set['K']*(ctr_fat**3)/8) & (input_set['N'] <= (input_set['K']+1)*(ctr_fat**3)/8) )
				cond_total = cond_total | (cond_loc & cond_sz & cond_fat)
				#set_row = input_set[cond_loc & cond_sz & cond_fat]
				#print (set_row)
			for ctr_thin in ctrs_thin:
				cond_thin = ((input_set['M'] == input_set['N']) & (input_set['N'] >= input_set['K']*8/(ctr_thin**3)) & (input_set['N'] <= (input_set['K']+1)*8/(ctr_thin**3)) )
				cond_total = cond_total | (cond_loc & cond_sz & cond_thin)
				#set_row = input_set[cond_loc & cond_sz & cond_thin]
				#print (set_row)
	print( "validation_set_split_BLAS3: %d pairs in the clean validation set" %( len(input_set[cond_total])))
	return input_set[cond_total]

def validation_set_split_BLAS1(name,locs,sizes,input_set):
	cond_total = False
	for loc in locs:
		cond_loc = ((input_set['Aloc'] ==  loc[0] ) & (input_set['Bloc'] ==  loc[1]) & (input_set['Cloc'] ==  -1))
		for size in sizes:
			cond_sz = (input_set['N'] == size)
			cond_total = cond_total | (cond_loc & cond_sz)
	print( "validation_set_split_BLAS1: %d pairs in the clean validation set" %( len(input_set[cond_total])))
	return input_set[cond_total]

def create_statistics(validation_set, val_col, pred_col_mine, pred_col_other):
		#print(merged.head(1))
		validation_set['PE_Mine'] = 100*(validation_set[pred_col_mine] - validation_set[val_col])/ validation_set[val_col]
		validation_set['PE_Comparisson'] = 100*(validation_set[pred_col_other] - validation_set[val_col])/ validation_set[val_col]
		print( "My MAPE : %lf" % abs(validation_set['PE_Mine']).mean())
		print( "Comparisson MAPE : %lf" % abs(validation_set['PE_Comparisson']).mean())

		print( "My PE Standard deviation : %lf" % validation_set['PE_Mine'].std())
		print( "Comparisson PE Standard deviation : %lf" % validation_set['PE_Comparisson'].std())

		#merged_clean = merged[((merged['PE_CoCopeLia'] <= 50) & (merged['PE_CoCopeLia'] >= -50)) & ((merged['PE_werkhoven'] <= 50) & (merged['PE_werkhoven'] >= -50))]
		my_outliers = validation_set[((validation_set['PE_Mine'] > 50) | (validation_set['PE_Mine'] < -50)) ]
		comparisson_outliers = validation_set[((validation_set['PE_Comparisson'] > 50) | (validation_set['PE_Comparisson'] < -50)) ]
		print( "Found %d Comparisson outliers:"	%( len(comparisson_outliers)))
		#print(comparisson_outliers)
		print( "Found %d CoCopeLia outliers:"	%( len(my_outliers)))
		#print(my_outliers)

		perper_mine = []
		perper_comp = []
		perper_static_sz = [1024,2048,3072,4096]
		perper_static_time = [[],[],[],[]]

		teams = validation_set.groupby(['M', 'N' , 'K', 'Aloc', 'Bloc', 'Cloc'])
		for state, curr_set in teams:
			#print(f"First 2 entries for {state!r}")
			#print("------------------------")
			#print(curr_set.head(2), end="\n\n")
			#curr_set = validation_set[(validation_set['M'] == size) & (validation_set['N'] == size) & (validation_set['K'] == size) & (validation_set['Aloc'] == loc[0]) & (validation_set['Bloc'] == loc[1]) & (validation_set['Cloc'] == loc[2])]
			perper_mine.append(curr_set[val_col].min()/curr_set.iloc[curr_set[pred_col_mine].argmin()][val_col])
			perper_comp.append(curr_set[val_col].min()/curr_set.iloc[curr_set[pred_col_other].argmin()][val_col])
			for static_sz_ctr in range(len(perper_static_sz)):
				static_sz = perper_static_sz[static_sz_ctr]
				if (curr_set[curr_set['T'] == static_sz].empty == False):
					perper_static_time[static_sz_ctr].append(float(curr_set[val_col].min()/curr_set[curr_set['T'] == static_sz][val_col]))
				else:
					perper_static_time[static_sz_ctr].append(float(curr_set[val_col].min()/curr_set[curr_set['T'] == curr_set['T'].max()][val_col]))
		if(len(teams)):
			perper_mine_geo = 100*stats.gmean(perper_mine,axis=0)
			perper_comp_geo = 100*stats.gmean(perper_comp,axis=0)
		print( "My Prediction Perf achieved (GEO.MEAN): %lf" % perper_mine_geo)
		print( "Comparisson Prediction Perf achieved (GEO.MEAN): %lf" % perper_comp_geo)

		for static_sz_ctr in range(len(perper_static_sz)):
			static_sz = perper_static_sz[static_sz_ctr]
			if len(perper_static_time[static_sz_ctr])!=0:
				perper_static_geo = 100*stats.gmean(perper_static_time[static_sz_ctr],axis=0)
				print( "Static T=%d Perf achieved for %d cases: %lf" % (static_sz, len(perper_static_time[static_sz_ctr]), perper_static_geo))		
		print("\n")
def cleanT(df, T_remove_list):
	apply_cond = True
	for T in T_remove_list:
		apply_cond = apply_cond & (df['T'] != T)
	return df[apply_cond]

