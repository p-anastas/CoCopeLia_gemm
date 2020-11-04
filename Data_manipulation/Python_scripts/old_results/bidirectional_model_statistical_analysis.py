import pandas as pd
import math

def GigaVal_per_s(val, time):
	return val * 1e-9 / time

def dgemm_flops(M,N,K):
	return M*N*(2*K+1)

machine_names= ['dungani','silver1']


def transfer_h2d(bytes,h2d_ti, h2d_tb):
	return h2d_ti + bytes* h2d_tb

def transfer_d2h(bytes,d2h_ti, d2h_tb):
	return d2h_ti + bytes* d2h_tb

for func in funcs:
	for machine in machine_names:

		validata = pd.read_csv('../Results/%s/validation/CoCoBLAS_%s_tile_0.log' % (machine, func), header =None, usecols = [0,1,2,3,5,6,7,12,13,14,15], names= ['M','N','K','T', 'Aloc','Bloc', 'Cloc', 'exec_t','streamed_t', 'cublasXt_t','BLASX_t' ])
		#print(validata.head(1))
		#print(validata['streamed_t'].max())
		pred_data = pd.read_csv('../Results/%s/predictions/%s_CoCopelia_predict_lr_0.log ' % (machine, func), header =None, usecols = [0,1,2,3,5,6,7,8,9], names= ['M','N','K','T', 'Aloc','Bloc', 'Cloc', 'werkhoven', 'CoCopelia'])
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
		merged_full = pd.merge(validata,pred_data, on = ['M', 'N', 'K', 'T', 'Aloc', 'Bloc', 'Cloc'])
		merged = merged_full#[(merged_full['N'] > 4096)]
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
		teams = merged.groupby(['M', 'N' , 'K', 'Aloc', 'Bloc', 'Cloc'])
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

			if func == 'sgemm' :
				dsize = 4
			else:
				dsize = 8
			temp = team[team['T'] == 2048]['exec_t']*(team[team['T'] == 2048]['M']/2048)**2 + \
			team[team['T'] == 2048]['Aloc'] * transfer_h2d(team[team['T'] == 2048]['M']* team[team['T'] == 2048]['K']*dsize,h2d_ti, h2d_tb) + \
			team[team['T'] == 2048]['Bloc'] * transfer_h2d(team[team['T'] == 2048]['N'] * team[team['T'] == 2048]['K']*dsize, h2d_ti, h2d_tb) +\
			team[team['T'] == 2048]['Cloc'] * transfer_h2d(team[team['T'] == 2048]['M'] * team[team['T'] == 2048]['N']*dsize, h2d_ti, h2d_tb) +\
			team[team['T'] == 2048]['Cloc'] * transfer_d2h(team[team['T'] == 2048]['M'] * team[team['T'] == 2048]['N']*dsize, d2h_ti, d2h_tb)
			#print(float(temp))
			if(int(team[team['T'] == 2048]['M']) == 16384 and int(team[team['T'] == 2048]['N']) == 16384 and \
					int(team[team['T'] == 2048]['K']) == 16384 and int(team[team['T'] == 2048]['Aloc']) == 1 and \
					int(team[team['T'] == 2048]['Bloc']) == 1 and int(team[team['T'] == 2048]['Cloc']) == 1):
					out_serial = team[team['T'] == 8192]['exec_t']*4 + transfer_h2d(3*16384*16384*8,h2d_ti, h2d_tb) + transfer_d2h(16384*16384*8, d2h_ti, d2h_tb)
					print("Serial MF t : %lf" % out_serial)
			serial_sl += 100 * abs(team['streamed_t'].min() / float(temp))
		print("CoCopelia pred sl: %lf" % (pred_sl_coco/len(teams)))
		print("Werkhoven pred sl: %lf" % (pred_sl_werk/len(teams)))
		print("Serial sl: %lf" % (serial_sl / len(teams)))

		merged['gflops'] = GigaVal_per_s(dgemm_flops(merged['M'],merged['N'],merged['K']),merged['streamed_t'])
		#merged['SE_CoCoBLAS'] = (merged['gflops'] -
								 #GigaVal_per_s(dgemm_flops(merged['M'],merged['N'],merged['K']),merged['CoCopelia']))**2
		#print(merged['SE_CoCoBLAS'])
		#merged['SE_werkhoven'] = (merged['gflops'] -
								 #GigaVal_per_s(dgemm_flops(merged['M'],merged['N'],merged['K']),merged['werkhoven']))**2
		#RMSE_coco = math.sqrt(merged['SE_CoCoBLAS'].mean()/merged['gflops'].mean())
		#RMSE_werk = math.sqrt(merged['SE_werkhoven'].mean()/merged['gflops'].mean())
		#print("CoCopelia RMSE: %lf" % (RMSE_coco))
		#print("Werkhoven RMSE: %lf" % (RMSE_werk))