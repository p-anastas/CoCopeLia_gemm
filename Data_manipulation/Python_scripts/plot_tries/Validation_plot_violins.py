import seaborn as sns
sns.set(style="whitegrid")
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import matplotlib as mpl
mpl.use('pdf')
import matplotlib.pyplot as plt
from pylab import cm

def GigaVal_per_s(val, time):
	return val * 1e-9 / time

def dgemm_flops(M,N,K):
	return M*N*(2*K+1)

font=8
plt.rc('font', family='serif', serif='Times')
#plt.rc('text', usetex=True)
plt.rc('xtick', labelsize=font)
plt.rc('ytick', labelsize=font)
plt.rc('axes', labelsize=font)

# width as measured in inkscape
width = 3.487
height = width / 1.618

fig, ax = plt.subplots()
#fig.subplots_adjust(left=.15, bottom=.16, right=.99, top=.97)
fig.subplots_adjust(left=.15, bottom=.17, right=.99, top=.97)


# Generate colors from the 'viridis' colormap
colors = cm.get_cmap('viridis',  4)
colors1 = cm.get_cmap('magma',  4)
color_mine = colors(0)
color_cublasxt = colors(2)
color_ideal = colors(3)
color_werk = colors1(2)

def validation_input(rootdir,machine,func,version):
	validata_CoCo = pd.read_csv('%s/Results/%s/validation/CoCopeLia_%s_0_v%s.log' % (rootdir, machine, func, version),
							   header =None, usecols = [0,1,2,3,4,5,6,8], names= ['M','N','K','T', 'Aloc','Bloc', 'Cloc',  'CoCopeLia_t'])
	validata_cuBLASXt = pd.read_csv('%s/Results/%s/validation/cuBLASXt_%s_0.log' % (rootdir, machine, func),
							   header =None, usecols = [0,1,2,3,4,5,6,8], names= ['M','N','K','T', 'Aloc','Bloc', 'Cloc', 'cublasXt_t'])

	print( "validation_input : Read %d values from \"%s/Results/%s/validation/CoCopeLia_%s_0_v%s.log\"..." %( len(validata_CoCo), rootdir,machine, func, version))
	print( "validation_input : Read %d values from \"%s/Results/%s/validation/cuBLASXt_%s_0.log\".." %( len(validata_cuBLASXt), rootdir,machine, func))

	validata = pd.merge(validata_CoCo, validata_cuBLASXt, on = ['M', 'N', 'K', 'T', 'Aloc', 'Bloc', 'Cloc'])
	#print(validata.head(1))
	#print(validata['CoCopeLia_t'].min())

	pred_data = pd.read_csv('%s/Results/%s/validation/%s_CoCopelia_predict_avg_0_v%s.log' % (rootdir,machine, func, version),
								header =None,  names= ['M','N','K','T', 'Aloc','Bloc', 'Cloc', 'werkhoven', 'CoCopelia'], dtype={'werkhoven': np.float64, 'CoCopelia': np.float64} ) #usecols = [0,1,2,3,5,6,7,8,9],

	print( "validation_input : Read %d predictions from \"%s/Results/%s/validation/%s_CoCopelia_predict_avg_0.log\".." %( len(pred_data), rootdir, machine, func))

	#print(pred_data.head(1))
	#print(pred_data['CoCopelia'].max())

	merged_full = pd.merge(validata,pred_data, on = ['M', 'N', 'K', 'T', 'Aloc', 'Bloc', 'Cloc'])

	print( "validation_input : Combined %d prediction/validation pairs" %( len(merged_full)))
	merged = merged_full[(merged_full['T'] >= 512) &  (merged_full['M']/1.5 > merged_full['T']) & (merged_full['N']/1.5 > merged_full['T']) & (merged_full['K']/1.5 > merged_full['T'])]
					#(merged_full['Aloc'] ==  1 ) & (merged_full['Bloc'] ==  1) & (merged_full['Cloc'] ==  1) & 	

	return merged

def validation_set_split(name,locs,mid_sizes,ctrs_fat,ctrs_thin,input_set):
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

machine_names= ['testbed-II_Tesla-V100']
funcs=['Dgemm']
version='1.2'
dim=[4096,4096,4096]
loc=[1,1,1]
for func in funcs:
	for machine in machine_names:

		print('Function: %s Machine : %s' % (func, machine))
		validation_mixed = validation_input("..",machine,func,version)
		print( "Main: %d pairs in the combined validation set" %( len(validation_mixed)))
		
		# Validation set 1 : Square dims, all locs
		locs= [[1,1,1], [1,1,0], [1,0,0], [1,0,1], [0,1,1], [0,1,0], [0,0,1]]
		mid_sizes= [4096,8192,12288,16384]
		ctrs_fat = [2]#[2,4,5,6,7,8]
		ctrs_thin = [] #[5,7,10,13,16]
		merged = validation_set_split("Vset1",locs,mid_sizes,ctrs_fat,ctrs_thin,validation_mixed)
		print( "Main: %d pairs in the square validation set" %( len(merged)))


		#print(merged.head(1))
		merged['PE_CoCopeLia'] = 100*(merged['CoCopeLia_t'] - merged['CoCopelia'])/ merged['CoCopeLia_t']
		merged['PE_werkhoven'] = 100*(merged['CoCopeLia_t'] - merged['werkhoven'])/ merged['CoCopeLia_t']
		print( "CoCopelia MAPE : %lf" % abs(merged['PE_CoCopeLia']).mean())
		print( "Werkhoven MAPE : %lf" % abs(merged['PE_werkhoven']).mean())

		print( "CoCopelia PE Standard deviation : %lf" % merged['PE_CoCopeLia'].std())
		print( "Werkhoven PE Standard deviation : %lf" % merged['PE_werkhoven'].std())

		#merged_clean = merged[((merged['PE_CoCopeLia'] <= 50) & (merged['PE_CoCopeLia'] >= -50)) & ((merged['PE_werkhoven'] <= 50) & (merged['PE_werkhoven'] >= -50))]
		merged_CoCo_outliers = merged[((merged['PE_CoCopeLia'] > 50) | (merged['PE_CoCopeLia'] < -50)) ]
		merged_werk_outliers = merged[((merged['PE_werkhoven'] > 50) | (merged['PE_werkhoven'] < -50)) ]

		merged['PE_CoCopeLia'] = (merged['CoCopeLia_t'] - merged['CoCopelia'])/ merged['CoCopeLia_t']
		merged['PE_werkhoven'] = (merged['CoCopeLia_t'] - merged['werkhoven'])/ merged['CoCopeLia_t']
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
		#print( "CoCopelia MAPE : %lf" % merged['APE_CoCopeLia'].mean())
		#print( "Werkhoven MAPE : %lf" % merged['APE_werkhoven'].mean())
		#teams = merged.groupby(['M', 'N' , 'K', 'Aloc', 'Bloc', 'Cloc'])
		#print(teams.head(1))
		x_axis = "T"

		merged['Sum_loc'] = merged['Aloc'] + merged['Bloc'] + merged['Cloc']

		merged_werk = merged
		merged_CoCo = merged

		merged_werk['PE'] = merged_werk['PE_werkhoven']
		merged_werk['model'] = 'Werkhoven'

		merged_CoCo['PE'] = merged_CoCo['PE_CoCopeLia']
		merged_werk['model'] = 'CoCopeLia'

		merged_uni = pd.concat({'': merged_werk, '': merged_CoCo})
		print(merged_uni.head(1))
		#sns.violinplot(x=x_axis,y="PE_CoCopeLia",data=merged, inner="box", palette="Set3", cut=2, label = "CoCopeLia Model") #linewidth=3, 
		#sns.violinplot(x=x_axis,y="PE_werkhoven",data=merged, inner="box", palette="Set3", cut=2, linewidth=3, label = "Werkhoven Model")
		sns_plot = sns.catplot(data=merged_uni, x=x_axis, y="PE", hue="keys", col = "M", kind="violin" ,height=4) #, split=True,height=4, aspect=.7
		sns.despine(left=True)
		sns_plot.savefig('Validation_plot_violins_sys-%s_func-%s_Xaxis-%s.pdf' % (func, machine,x_axis) )
		#plt.legend(fontsize=font, loc='upper center', fancybox = False, ncol=3)

		ax.set_ylabel('Percentile error')
		ax.set_xlabel('Problem dim(M)')

		#fig.set_size_inches(width, height)
		#fig.savefig('Validation_plot_violins_sys-%s_func-%s_Xaxis-%s.pdf' % (func, machine,x_axis) )

