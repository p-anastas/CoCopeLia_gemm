import pandas as pd
import math

def GigaVal_per_s(val, time):
	return val * 1e-9 / time

def dgemm_flops(M,N,K):
	return M*N*(2*K+1)

machine_names= ['silver1'] #'dungani',
funcs=['daxpy']

def transfer_h2d(bytes,h2d_ti, h2d_tb):
	return h2d_ti + bytes* h2d_tb

def transfer_d2h(bytes,d2h_ti, d2h_tb):
	return d2h_ti + bytes* d2h_tb

for func in funcs:
	for machine in machine_names:

		validata = pd.read_csv('../Results/%s/validation/CoCoBLAS_%s_tile_0.log' % (machine, func), header =None,
							   usecols = [0,1,2,3,7,8,9], names= ['N','Ns', 'xloc','yloc', 'exec_t','streamed_t', 'uni_t'])
		#print(validata.head(1))
		#print(validata['streamed_t'].max())

		pred_data = pd.read_csv('../Results/%s/predictions/%s_CoCopelia_predict_lr_0.log' % (machine, func), header=None,
								   usecols=[0, 1, 2, 3, 4,5],
								   names=['N', 'Ns', 'xloc', 'yloc', 'werkhoven', 'CoCopelia'])
		#print(pred_data.head(1))
		#print(pred_data['CoCopelia'].max())

		infile_h2d = '../Results/%s/Models/transfer_model_0_-1.log' % machine
		infile_d2h = '../Results/%s/Models/transfer_model_-1_0.log' % machine

		with open(infile_h2d, 'r') as infile:
			h2d_log = infile.readlines()

		with open(infile_d2h, 'r') as infile:
			d2h_log = infile.readlines()

		h2d_ti = float(h2d_log[0])
		h2d_tb = float(h2d_log[1])
		# h2d_sl.append (float(h2d_log[2]))
		# print('ti = %e, tb = %e, sl = %lf' % (h2d_ti, h2d_tb, h2d_sl))

		d2h_ti = float(d2h_log[0])
		d2h_tb = float(d2h_log[1])
		# d2h_sl.append (float(d2h_log[2]))
		# print('ti = %e, tb = %e, sl = %lf' % (d2h_ti, d2h_tb, d2h_sl))
		merged_full = pd.merge(validata, pred_data, on=['N', 'Ns', 'xloc', 'yloc'])
		merged = merged_full[(merged_full['N'] >= 8388608)]# & (merged_full['Ns'] >= merged_full['N']/64) & (merged_full['xloc'] == 1) & (merged_full['yloc'] == 1)]
		#print(merged)
		merged['APE_CoCoBLAS'] = 100*abs(merged['streamed_t'] - merged['CoCopelia'])/ merged['streamed_t']
		merged['APE_werkhoven'] = 100*abs(merged['streamed_t'] - merged['werkhoven'])/ merged['streamed_t']
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
		print( "CoCopelia MAPE : %lf" % merged['APE_CoCoBLAS'].mean())
		print( "Werkhoven MAPE : %lf" % merged['APE_werkhoven'].mean())
		teams = merged.groupby(['N', 'xloc', 'yloc'])
		pred_sl_coco = 0
		pred_sl_werk = 0
		serial_sl = 0
		for name,team in teams:
			#pred_sl_coco += 100*abs((team.iloc[team['CoCopelia'].argmin()]['streamed_t'] -
									 #team['streamed_t'].min())/team['streamed_t'].min())
			pred_sl_coco += 100 * abs( team['streamed_t'].min()/ team.iloc[team['CoCopelia'].argmin()]['streamed_t'])
			#pred_sl_werk += 100 * abs((team.iloc[team['werkhoven'].argmin()]['streamed_t'] -
			 #team['streamed_t'].min()) / team['streamed_t'].min())
			pred_sl_werk += 100 * abs(team['streamed_t'].min() / team.iloc[team['werkhoven'].argmin()]['streamed_t'])
			temp = team[team['Ns'] == team['N']/2]['exec_t'] * 2+ \
			team[team['Ns'] == team['N']/2]['xloc'] * transfer_h2d(team[team['Ns'] == team['N']/2]['N']*8, h2d_ti, h2d_tb) + \
			team[team['Ns'] == team['N']/2]['yloc'] * transfer_h2d(team[team['Ns'] == team['N']/2]['N']*8, h2d_ti, h2d_tb) +\
			team[team['Ns'] == team['N']/2]['yloc'] * transfer_d2h(team[team['Ns'] == team['N']/2]['N']*8, d2h_ti, d2h_tb)
			#print(temp)
			serial_sl += 100 * abs(team['streamed_t'].min() / float(temp))
		print("CoCopelia pred sl: %lf" % (pred_sl_coco/len(teams)))
		print("Werkhoven pred sl: %lf" % (pred_sl_werk/len(teams)))
		print("Serial sl: %lf" % (serial_sl / len(teams)))
