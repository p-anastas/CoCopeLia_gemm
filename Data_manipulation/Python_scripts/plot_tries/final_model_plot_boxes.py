import seaborn as sns
sns.set(style="whitegrid")
import pandas as pd
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

machine_names= ['testbed-II_Tesla-V100']
funcs=['Dgemm']
version='1.1'
dim=[4096,4096,4096]
loc=[1,1,1]
mindim=768
maxdim=min(min(dim),8192,max(dim)/1.5) # min(dim)/2
for func in funcs:
	for machine in machine_names:

		validata_CoCo = pd.read_csv('../Results/%s/validation/CoCopeLia_%s_0_v%s.log' % (machine, func, version),
							   header =None, usecols = [0,1,2,3,4,5,6,8], names= ['M','N','K','T', 'Aloc','Bloc', 'Cloc',  'CoCopeLia_t'])
		validata_cuBLASXt = pd.read_csv('../Results/%s/validation/cuBLASXt_%s_0.log' % (machine, func),
							   header =None, usecols = [0,1,2,3,4,5,6,8], names= ['M','N','K','T', 'Aloc','Bloc', 'Cloc', 'cublasXt_t'])

		validata = pd.merge(validata_CoCo, validata_cuBLASXt, on = ['M', 'N', 'K', 'T', 'Aloc', 'Bloc', 'Cloc'])
		
		#print(validata.head(1))
		#print(validata['CoCopeLia_t'].min())

		pred_data_avg = pd.read_csv('../Results/%s/validation/%s_CoCopelia_predict_avg_0.log' % (machine, func),
								header =None,  names= ['M','N','K','T', 'Aloc','Bloc', 'Cloc', 'werkhoven', 'CoCopelia'])

		pred_data_min = pd.read_csv('../Results/%s/validation/%s_CoCopelia_predict_min_0.log' % (machine, func),
								header =None,  names= ['M','N','K','T', 'Aloc','Bloc', 'Cloc', 'werkhoven', 'CoCopelia'])

		pred_data_max = pd.read_csv('../Results/%s/validation/%s_CoCopelia_predict_max_0.log' % (machine, func),
								header =None,  names= ['M','N','K','T', 'Aloc','Bloc', 'Cloc', 'werkhoven', 'CoCopelia'])

		#pred_data = pred_data_avg

		#print(pred_data.head(1))
		#print(pred_data['CoCopelia'].max())

		merged_full = pd.merge(validata,pred_data_avg, on = ['M', 'N', 'K', 'T', 'Aloc', 'Bloc', 'Cloc'])
		validata['size'] = validata['M']*validata['N']*validata['K']
		merged = merged_full[((validata['K'] != 4096) | (validata['M'] == 32768)) &
					(validata['size'] < validata['T']**3)]# &
					#(validata['T'] > 768) &
		#(validata['M'] == dim[0]) & (validata['N'] == dim[1]) & (validata['K'] == dim[2]) & 
					#(merged_full['Aloc'] == loc[0]) & (merged_full['Bloc'] == loc[1]) & (merged_full['Cloc'] == loc[2])]
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
		teams = merged.groupby(['M', 'N' , 'K', 'Aloc', 'Bloc', 'Cloc'])
		#print(teams.head(1))
		sns.violinplot(x="M",y="PE_CoCopeLia",data=merged, inner="box", palette="Set3", cut=2, linewidth=3, label = "CoCopeLia Model")
		#sns.violinplot(x="M",y="PE_werkhoven",data=merged, inner="box", palette="Set3", cut=2, linewidth=3, label = "Werkhoven Model")
		sns.despine(left=True)

		#plt.legend(fontsize=font, loc='upper center', fancybox = False, ncol=3)

		ax.set_ylabel('Percentile error')
		ax.set_xlabel('Problem dim(M)')

		#fig.set_size_inches(width, height)
		fig.savefig('final_model_boxes_CoCopeLia_%s_%s.pdf' % (func, machine) )

