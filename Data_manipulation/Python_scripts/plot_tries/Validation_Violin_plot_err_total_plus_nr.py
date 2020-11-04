import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import math
import numpy as np
from collections import OrderedDict

import matplotlib as mpl
mpl.use('pdf')
import matplotlib.pyplot as plt
from pylab import cm

def validation_input(rootdir,machine,func,version):
	if func!= 'Daxpy':
		validata_CoCo = pd.read_csv('%s/Results/%s/validation/CoCopeLia_%s_0_v%s.log' % (rootdir, machine, func, version),
				header =None, usecols = [0,1,2,3,4,5,6,8], names= ['M','N','K','T', 'Aloc','Bloc', 'Cloc',  'CoCopeLia_t'])
		validata_cuBLASXt = pd.read_csv('%s/Results/%s/validation/cuBLASXt_%s_0.log' % (rootdir, machine, func),
				header =None, usecols = [0,1,2,3,4,5,6,8], names= ['M','N','K','T', 'Aloc','Bloc', 'Cloc', 'cublasXt_t'])

		print( "validation_input : Read %d values from \"%s/Results/%s/validation/CoCopeLia_%s_0_v%s.log\"..." %( len(validata_CoCo), rootdir,machine, func, version))
		print( "validation_input : Read %d values from \"%s/Results/%s/validation/cuBLASXt_%s_0.log\".." %( len(validata_cuBLASXt), rootdir,machine, func))

		validata = pd.merge(validata_CoCo, validata_cuBLASXt, on = ['M', 'N', 'K', 'T', 'Aloc', 'Bloc', 'Cloc'])

		pred_data_reuse = pd.read_csv('%s/Results/%s/validation/%s_CoCopelia_predict_avg_0_v%s.log' % (rootdir,machine, func, version),
				header =None,  names= ['M','N','K','T', 'Aloc','Bloc', 'Cloc', 'werkhoven', 'CoCopelia'], dtype={'werkhoven': np.float64, 'CoCopelia': np.float64} ) #usecols = [0,1,2,3,5,6,7,8,9],

		pred_data_no_reuse = pd.read_csv('%s/Results/%s/validation/%s_CoCopelia_predict_no_reuse_0_v%s.log' % (rootdir,machine, func, version),
				header =None,  names= ['M','N','K','T', 'Aloc','Bloc', 'Cloc', 'werkhoven', 'CoCopelia_nr'], dtype={'werkhoven': np.float64, 'CoCopelia_nr': np.float64} ) #usecols = [0,1,2,3,5,6,7,8,9],

		pred_data = pd.merge(pred_data_reuse,pred_data_no_reuse, on = ['M', 'N', 'K', 'T', 'Aloc', 'Bloc', 'Cloc', 'werkhoven'])

	else: 
		validata = pd.read_csv('%s/Results/%s/validation/CoCopeLia_%s_0_v%s.log' % (rootdir, machine, func, version),
							   header =None, usecols = [0,1,2,3,8,9], names= ['N', 'T', 'Aloc','Bloc', 'CoCopeLia_t', 'unified_t'])
		print( "validation_input : Read %d values from \"%s/Results/%s/validation/CoCopeLia_%s_0_v%s.log\"..." %( len(validata), rootdir,machine, func, version))

		pred_data = pd.read_csv('%s/Results/%s/validation/%s_CoCopelia_predict_avg_0_v%s.log' % (rootdir,machine, func, version),
				header =None,  names= ['N','T', 'Aloc','Bloc', 'werkhoven', 'CoCopelia'], dtype={'werkhoven': np.float64, 'CoCopelia': np.float64} ) #usecols = [0,1,2,3,5,6,7,8,9],

		validata['M'] = validata['K'] = pred_data['M'] = pred_data['K'] = validata['Cloc'] = pred_data['Cloc'] = -1

	print( "validation_input : Read %d predictions from \"%s/Results/%s/validation/%s_CoCopelia_predict_avg_0.log\".." %( len(pred_data), rootdir, machine, func))

	#print(validata.head(1))
	#print(validata['CoCopeLia_t'].min())
	#print(pred_data.head(1))
	#print(pred_data['CoCopelia'].max())

	merged_full = pd.merge(validata,pred_data, on = ['M', 'N', 'K', 'T', 'Aloc', 'Bloc', 'Cloc'])

	print( "validation_input : Combined %d prediction/validation pairs" %( len(merged_full)))
	if func!= 'Daxpy':
		merged = merged_full[(merged_full['T'] >= 512) &  (merged_full['M']/1.5 > merged_full['T']) & (merged_full['N']/1.5 > merged_full['T']) & (merged_full['K']/1.5 > merged_full['T'])]
	else:
		merged = merged_full[(merged_full['T'] >= merged_full['N']/64) &  (merged_full['T'] < merged_full['N']/1.5)]
					#(merged_full['Aloc'] ==  1 ) & (merged_full['Bloc'] ==  1) & (merged_full['Cloc'] ==  1) & 	

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

	return input_set[cond_total]

def validation_set_split_BLAS1(name,locs,sizes,input_set):
	cond_total = False
	for loc in locs:
		cond_loc = ((input_set['Aloc'] ==  loc[0] ) & (input_set['Bloc'] ==  loc[1]) & (input_set['Cloc'] ==  -1))
		for size in sizes:
			cond_sz = (input_set['N'] == size)
			cond_total = cond_total | (cond_loc & cond_sz)

	return input_set[cond_total]

machine_names= ['testbed-I_Tesla-K40', 'testbed-II_Tesla-V100'] #
cool_names= ['testbed-I', 'testbed-II'] 
funcs=['Dgemm','Sgemm','Daxpy'] # 
version='1.2'

font=8
plt.rc('font', family='serif', serif='Times')
#plt.rc('text', usetex=True)
plt.rc('xtick', labelsize=font-1)
plt.rc('ytick', labelsize=font-1)
plt.rc('axes', labelsize=font)


# width as measured in inkscape
width = 3.487
height = width / 1.618*3/2

fig, axs = plt.subplots( len(funcs), len(machine_names), sharey=True)
#fig.subplots_adjust(left=.15, bottom=.16, right=.99, top=.97)
fig.subplots_adjust(left=.15, bottom=.08, right=.99, top=.94)


# Generate colors from the 'viridis' colormap
colors = cm.get_cmap('viridis',  4)
colors1 = cm.get_cmap('magma',  4)
color_mine = colors(0)
color_cublasxt = colors(2)
color_ideal = colors(3)
color_werk = colors1(2)

import seaborn as sns
sns.set(style="whitegrid", palette=[color_werk,color_cublasxt],font_scale=0.2, rc={"lines.linewidth": 1})


ctr = 0
for machine in machine_names:
	functr = 0
	for func in funcs:
		print('Function: %s Machine : %s' % (func, machine))
		validation_mixed = validation_input("..",machine,func,version)
		print( "Main: %d pairs in the combined validation set" %( len(validation_mixed)))
		
		# Validation set 1 : Square dims, all locs
		if func!= 'Daxpy':
			locs= [[1,1,1], [1,1,0], [1,0,0], [1,0,1], [0,1,1], [0,1,0], [0,0,1]]
			mid_sizes= [4096,8192,12288,16384]
			ctrs_fat = [2]#[2,4,5,6,7,8]
			ctrs_thin = [] #[5,7,10,13,16]
			merged = validation_set_split_BLAS3("Vset1",locs,mid_sizes,ctrs_fat,ctrs_thin,validation_mixed)
		else: 
			locs= [[1,1], [1,0], [0,1]]
			sizes= list(map(lambda y: y*1024*1024, [8,64,128,256]))
			print(sizes)
			merged = validation_set_split_BLAS1("Vset1",locs,sizes,validation_mixed)		
		print( "Main: %d pairs in the square validation set" %( len(merged)))


		#print(merged.head(1))
		merged['PE_CoCopeLia'] = 100*(merged['CoCopeLia_t'] - merged['CoCopelia'])/ merged['CoCopeLia_t']
		if func !='Daxpy':
			merged['PE_CoCopeLia_nr'] = 100*(merged['CoCopeLia_t'] - merged['CoCopelia_nr'])/ merged['CoCopeLia_t']
		merged['PE_werkhoven'] = 100*(merged['CoCopeLia_t'] - merged['werkhoven'])/ merged['CoCopeLia_t']
		print( "CoCopelia MAPE : %lf" % abs(merged['PE_CoCopeLia']).mean())
		if func !='Daxpy':
			print( "CoCopelia_nr MAPE : %lf" % abs(merged['PE_CoCopeLia_nr']).mean())
		print( "Werkhoven MAPE : %lf" % abs(merged['PE_werkhoven']).mean())

		print( "CoCopelia PE Standard deviation : %lf" % merged['PE_CoCopeLia'].std())
		if func !='Daxpy':
			print( "CoCopelia_nr PE Standard deviation : %lf" % merged['PE_CoCopeLia_nr'].std())
		print( "Werkhoven PE Standard deviation : %lf" % merged['PE_werkhoven'].std())

		#merged_clean = merged[((merged['PE_CoCopeLia'] <= 50) & (merged['PE_CoCopeLia'] >= -50)) & ((merged['PE_werkhoven'] <= 50) & (merged['PE_werkhoven'] >= -50))]
		merged_CoCo_outliers = merged[((merged['PE_CoCopeLia'] > 50) | (merged['PE_CoCopeLia'] < -50)) ]
		merged_CoCo_nr_outliers = merged[((merged['PE_CoCopeLia_nr'] > 50) | (merged['PE_CoCopeLia_nr'] < -50)) ]
		merged_werk_outliers = merged[((merged['PE_werkhoven'] > 50) | (merged['PE_werkhoven'] < -50)) ]
		#print( "Kept %d pairs in after cleaning final validation set from outliers (0.50)" %( len(merged_clean)))
		#print( "From %d Werkhoven outliers:"	%( len(merged_werk_outliers)))
		#print(merged_werk_outliers)
		#print( "From %d CoCopeLia outliers:"	%( len(merged_CoCo_outliers)))
		print(merged_CoCo_nr_outliers)
		#print( "CoCopelia MAPE without outliers (0.50): %lf" % abs(merged_clean['PE_CoCopeLia']).mean())
		#print( "Werkhoven MAPE without outliers (0.50): %lf" % abs(merged_clean['PE_werkhoven']).mean())
		print("\n")
		
		#teams = merged.groupby(['M', 'N' , 'K', 'Aloc', 'Bloc', 'Cloc'])

		#merged['PE_CoCoBLAS'] = 100*(merged['streamed_t'] - merged['CoCopelia'])/ merged['streamed_t']
		#merged['PE_werkhoven'] = 100*(merged['streamed_t'] - merged['werkhoven'])/ merged['streamed_t']
		#print(merged.iloc[merged['CoCopelia'].argmax()])
		#merged_under_co = merged[merged['PE_CoCoBLAS'] > 0]
		#merged_over_co = merged[merged['PE_CoCoBLAS'] <= 0]
		#merged_under_we = merged[merged['PE_werkhoven'] > 0]
		#merged_over_we = merged[merged['PE_werkhoven'] <= 0]
		#print( "CoCopelia MAPE : %lf, NMPE : %lf, PMPE : %lf" % (merged['APE_CoCoBLAS'].mean(),merged_under_co['PE_CoCoBLAS'].mean(),merged_over_co['PE_CoCoBLAS'].mean()))
		#print( "Werkhoven MAPE : %lf, NMPE : %lf, PMPE : %lf" % (merged['APE_werkhoven'].mean(),merged_under_we['PE_werkhoven'].mean(),merged_over_we['PE_werkhoven'].mean()))

		x_axis = 'func'

		merged_werk = merged.copy()
		merged_CoCo = merged.copy()
		merged_CoCo_nr = merged.copy()

		merged_werk['PE'] = merged_werk['PE_werkhoven']
		merged_werk['model'] = 'Werk. 2 cpe. [12]'

		merged_CoCo['PE'] = merged_CoCo['PE_CoCopeLia']
		merged_CoCo['model'] = 'CoCopeLia'

		if func !='Daxpy':
			merged_CoCo_nr['PE'] = merged_CoCo['PE_CoCopeLia_nr']
			merged_CoCo_nr['model'] = 'CoCopeLia_nr'
			merged_uni = pd.concat([merged_werk, merged_CoCo, merged_CoCo_nr])
		else:
			merged_uni = pd.concat([merged_werk, merged_CoCo])
		merged_uni['func'] = func
		print(merged_uni.head(1))
		axs[functr,ctr].axhline(0, ls='--', linewidth = 0.6, color = 'gray', alpha=.7)
		axs[functr,ctr].axhline(20, ls='--', linewidth = 0.6, color = 'gray', alpha=.7)
		axs[functr,ctr].axhline(-20, ls='--', linewidth = 0.6, color = 'gray', alpha=.7)
		#axs[functr,ctr].grid(ls='--', linewidth = 0.6, color = 'gray')
		sns.violinplot(x=x_axis,y="PE",data=merged_uni, hue ='model', inner="box", linewidth = 1, ax=axs[functr,ctr])
		axs[functr,ctr].get_legend().remove()
		axs[functr,ctr].set_xticks([])
		axs[functr,ctr].set_xlabel('')

		if (functr == 0):
			axs[functr,ctr].set_title(cool_names[ctr], fontsize=font)


		if (ctr == 0):
			#axs[functr,ctr].set_ylabel(func)
			axs[functr,ctr].text( 0.3, 50.0, func, fontsize=font)

		#axs[functr,ctr].set_ylabel(func)
		if ((functr == 1) & (ctr == 0)):
			axs[functr,ctr].set_ylabel("Prediction Error (%)", fontsize=font)
		#elif (ctr == 0):
			#axs[functr,ctr].set_ylabel(func)
		else:
			axs[functr,ctr].set_ylabel('')
		#if (functr == 2):
		#	axs[functr,ctr].set_xlabel(machine)
		#else:
		#	axs[functr,ctr].set_xlabel('')
		sns.despine()
		#sns_plot.savefig('Validation_plot_violins_sys-%s_func-%s_Xaxis-%s.pdf' % (func, machine,x_axis) )
		#



		functr += 1
	ctr += 1

#for ax in axs.flat:
#	ax.set(xlabel=machine_names, ylabel=funcs)

#Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
	ax.label_outer()

fig.set_size_inches(width, height)
# Create the legend
handles, labels = fig.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
fig.legend(by_label.values(), by_label.keys(),
	loc="lower center",   # Position of legend
	borderaxespad=0.1,    # Small spacing around legend box
	#title="Model",  # Title for the legend
	fontsize=font, fancybox = False, ncol=2
	)
fig.savefig('Validation_plot_violins_all_plus_nr_Xaxis-%s.pdf' % (x_axis) )


