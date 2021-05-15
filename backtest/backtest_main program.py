# -*- coding: utf-8 -*-
# Questions:
# 1 - when does this data become publicly available to trade on?
# 2- is this data 'as of' or was it later adjusted?
# 3 - why the big gaps in the data, and the big jumps in values?
# 4 - I'm missing the first available data point (2003 mid year, which could flow over into this dataset...)
"""
Created on Thu Oct 20 18:41:25 2016

@author: Richard
"""
############ IMPORTS
import pandas as pd
#import numpy as np
#import matplotlib.pyplot as plt
import os.path
#from scipy.stats import norm
import backtest as bt
#from scipy.stats.mstats import gmean

############ INPUTS
#cost = 0.000
frequency = 12.0

path=os.path.join('C:\Users','Richard','Dropbox','RJ & JVW','Backtest','csv dataset3')
#fp_len = 48 # number of months to go back in fp - default is 8 years or 96 months
inputs=[]
for filenames in os.listdir(path):
    if os.path.isfile(os.path.join(path, filenames)):
        just_name = os.path.splitext(filenames)
        inputs.append(just_name[0])

inputs_not_to_shift = ['price_monthly','volume_monthly','market_val']
inputs_to_shift=[item for item in inputs if item not in inputs_not_to_shift]

############ RUN CODE
# create a data object 'd' using backtest.py
d = bt.Data(path, inputs, inputs_to_shift, months_delay_data=3, start=0, delist_value=1)

#data_present = d.accruals()*d.pfd()*d.pman()*d.market_cap()*d.mom()*d.mvi_weight()*d.value()*d.fip()
#data_present = data_present.notnull().astype(int) # would be more reassuring if I had na's here..
#data_present = data_present.replace(0,np.nan)

init_pos1 = pd.DataFrame(1, index=d.basic_data['price_monthly'].index, columns=d.basic_data['price_monthly'].columns)
non_fin_sectors = list(pd.read_csv(os.path.join(path,'other data','non-fins.csv'))) #list(sectors.index[sectors==0])
non_fin_sectors.remove('WES-JSE') # there's a woopsie in the price data for this stock, so I've just excluded it for now...
init_pos_non_fins= init_pos1[non_fin_sectors]

############## INIT POSITION - JUST FIN STOCKS
fin_sectors = list(pd.read_csv(os.path.join(path,'other data','financials.csv'))) #list(sectors.index[sectors==0])
fin_sectors.remove('STP-JSE') # there's a woopsie in the price data for this stock, so I've just excluded it for now...
init_pos_fin= init_pos1[fin_sectors]

banks = list(pd.read_csv(os.path.join(path,'other data','banks.csv')))
#['DSY-JSE', 'LBH-JSE','MMI-JSE','OML-JSE','SLM-JSE','SNT-JSE']
insurers = list(pd.read_csv(os.path.join(path,'other data','insurers.csv')))
#['BGA-JSE','CPI-JSE','FSR-JSE','INL-JSE','INP-JSE','NED-JSE','RMH-JSE','RMI-JSE','SBK-JSE']

#init_pos1 = init_pos1 * data_present

def run_backtests():
    ret={}  
    mc=bt.Mkt_cap_scr(threshold=1000)
    mw=bt.Mkt_cap_weights()
    
    QM_object = bt.QM(threshold=1000, scr2_perc = 0.15, scr3_perc = 0.6, upper_limit=0.2)
    QM_object.backtest(init_pos= init_pos1, data = d) #, rebalance = 'M', #Q-NOV, Q-DEC, Q-OCT                        
    ret['QM'] = QM_object.calc_ret(price_data= d.basic_data['price_monthly'])
    
    QV_object = bt.QV(threshold=1000, scr2_perc=0.95, scr3_perc = 0.15, scr4_perc = 0.6, upper_limit=0.2)
    QV_object.backtest(init_pos= init_pos1, data=d) #, rebalance = 'M'                                  
    ret['QV'] = QV_object.calc_ret(price_data= d.basic_data['price_monthly'])
    
    ret['QM+QV'] = (QM_object.calc_ret(price_data= d.basic_data['price_monthly']) + QV_object.calc_ret(price_data= d.basic_data['price_monthly'])  )/2.0
 
    QV_fin_object = bt.QV_fin(banks, insurers, threshold=0, scr2_perc = 0.15, scr3_perc = 0.6, upper_limit=0.2)
    QV_fin_object.backtest(init_pos= init_pos_fin, data=d) #, rebalance = 'M',                                                             
    ret['QV_fin'] = QV_fin_object.calc_ret(price_data= d.basic_data['price_monthly'])
    
    BAH_object = bt.BAH()
    BAH_object.backtest(mc.run(init_pos1,data=d), data=d)
    ret['alsi'] = BAH_object.calc_ret(d.basic_data['price_monthly'])  

    mom_object = bt.Mom(scr_perc = 0.15)
    mom_object.run(mc.run(init_pos1,data=d), data=d)
    ret['mom'] = mom_object.calc_ret(d.basic_data['price_monthly'])  
    
    fip_object = bt.Fip(scr_perc = 0.6)
    fip_object.run(mc.run(init_pos1,data=d), data=d)
    ret['fip'] = fip_object.calc_ret(d.basic_data['price_monthly'])
    
    acc_object = bt.Fip(scr_perc = 0.95)
    acc_object.run(mc.run(init_pos1,data=d), data=d)
    ret['acc'] = acc_object.calc_ret(d.basic_data['price_monthly'])
    metrics=(['acc'])
    metrics=(['QV', 'QV_fin','mom','fip','QM','alsi']) #
                                
    bt.plot_returns(ret, metrics)    
    bt.tabulate_results(ret, metrics, frequency=12.0, risk_free = 0.07) # risk free rate is per annum
    bt.plot_CAGR(ret, metrics, 1)
    bt.plot_CAGR(ret, metrics, 5)

def QM_mkt_cap(init_pos = init_pos1, threshold = 1000, scr_perc_mom = 0.2, scr_perc_fip = 0.6):  
    mc=bt.Mkt_cap_scr(threshold = threshold)
    mom_object = bt.Mom(scr_perc = scr_perc_mom)
    fip_object = bt.Fip(scr_perc = scr_perc_fip)
    mw = bt.Mkt_cap_weights()
    
    mw.run(fip_object.run(mom_object.run(mc.run(init_pos,data=d), data=d), data=d),data=d)
    return mw

def QM_vmi(init_pos = init_pos1, threshold = 1000, scr_perc_mom = 0.2, scr_perc_fip = 0.6):
    mc=bt.Mkt_cap_scr(threshold = 1000)
    mom_object = bt.Mom(scr_perc = 0.2)
    fip_object = bt.Fip(scr_perc = 0.6)
    mw = bt.Mvi_weights()
    
    mw.run(fip_object.run(mom_object.run(mc.run(init_pos,data=d), data=d), data=d),data=d)
    return mw
    
def QM_parts():
    ret={}  
    #mc=bt.Mkt_cap_scr(threshold=1000)
    
    QM_object = bt.QM(threshold=1000, scr2_perc = 0.2, scr3_perc = 0.6, upper_limit=0.2)
    QM_object.backtest(init_pos= init_pos1, data = d) #, rebalance = 'M', #Q-NOV, Q-DEC, Q-OCT                        
    ret['QM'] = QM_object.calc_ret()
    
#    BAH_object = bt.BAH()
#    BAH_object.backtest(mc.run(init_pos1,data=d), data=d)
#    ret['alsi'] = BAH_object.calc_ret(d.basic_data['price_monthly'])  
#
#    mom_object = bt.Mom(scr_perc = 0.2)
#    mom_object.run(mc.run(init_pos1,data=d), data=d)
#    ret['mom'] = mom_object.calc_ret(d.basic_data['price_monthly'])  
#    
#    fip_object = bt.Fip(scr_perc = 0.6)
#    fip_object.run(mc.run(init_pos1,data=d), data=d)
#    ret['fip'] = fip_object.calc_ret(d.basic_data['price_monthly'])
    
    metrics=(['QM']) #,'fip','QM','alsi'
                                
    bt.plot_returns(ret, metrics)    
    bt.tabulate_results(ret, metrics, frequency=12.0, risk_free = 0.07) # risk free rate is per annum
    bt.plot_CAGR(ret, metrics, 1)
    bt.plot_CAGR(ret, metrics, 5)

# this runs all of the component parts of the QV strategies as separate sub-strategies    
def QV_parts():
    ret={}  
    mc=bt.Mkt_cap_scr(threshold=100)
    mw = bt.Mkt_cap_weights()
    #mw = bt.Mvi_weights()
    
    #QV_object = bt.QV(threshold=1000, scr2_perc=0.95, scr3_perc = 0.15, scr4_perc = 0.6, upper_limit=0.2)
    #QV_object.backtest(init_pos= init_pos1, data=d) #, rebalance = 'M'                                  
   #ret['QV'] = QV_object.calc_ret(price_data= d.basic_data['price_monthly'])
    
    #QV_fin_object = bt.QV_fin(banks, insurers, threshold=0, scr2_perc = 0.15, scr3_perc = 0.6, upper_limit=0.2)
    #QV_fin_object.backtest(init_pos= init_pos_fin, data=d) #, rebalance = 'M',                                                             
    #ret['QV_fin'] = QV_fin_object.calc_ret(price_data= d.basic_data['price_monthly'])
    
    BAH_object = bt.BAH()
    mw.run(BAH_object.backtest(mc.run(init_pos_non_fins,data=d), data=d), data=d)
    mw.final_positions = bt.limit_pos_size(mw.final_positions)
    ret['alsi'] = mw.calc_ret(d.basic_data['price_monthly'])  
    
    acc_object = bt.Accruals(scr_perc = 0.05)
    mw.run(acc_object.run(mc.run(init_pos_non_fins,data=d), data=d), data=d)
    mw.final_positions = bt.limit_pos_size(mw.final_positions)
    ret['acc'] = mw.calc_ret(d.basic_data['price_monthly'])
    
    pman_object = bt.Pman(scr_perc = 0.05)
    mw.run(pman_object.run(mc.run(init_pos_non_fins,data=d), data=d), data=d)
    mw.final_positions = bt.limit_pos_size(mw.final_positions)
    ret['pman'] = mw.calc_ret(d.basic_data['price_monthly'])
    
    pfd_object = bt.Pfd(scr_perc = 0.05)
    mw.run(pfd_object.run(mc.run(init_pos_non_fins,data=d), data=d), data=d)
    mw.final_positions = bt.limit_pos_size(mw.final_positions)
    ret['pfd'] = mw.calc_ret(d.basic_data['price_monthly'])
    
    for_object = bt.Forensic(scr_perc = 0.05)
    mw.run(for_object.run(mc.run(init_pos_non_fins,data=d), data=d), data=d)
    mw.final_positions = bt.limit_pos_size(mw.final_positions)
    ret['for'] = mw.calc_ret(d.basic_data['price_monthly'])
    
    fs_object = bt.Fs(scr_perc = 0.6)
    mw.run(fs_object.run(mc.run(init_pos_non_fins,data=d), data=d), data=d)
    mw.final_positions = bt.limit_pos_size(mw.final_positions)
    ret['fs'] = mw.calc_ret(d.basic_data['price_monthly'])
    
    pfp_object = bt.Pfp(scr_perc = 0.6)
    mw.run(pfp_object.run(mc.run(init_pos_non_fins,data=d), data=d), data=d)
    mw.final_positions = bt.limit_pos_size(mw.final_positions)
    ret['pfp'] = mw.calc_ret(d.basic_data['price_monthly'])
    
    qual_object = bt.Quality(scr_perc = 0.6)
    mw.run(qual_object.run(mc.run(init_pos_non_fins,data=d), data=d), data=d)
    mw.final_positions = bt.limit_pos_size(mw.final_positions)
    ret['qual'] = mw.calc_ret(d.basic_data['price_monthly'])
    
    value_object = bt.Value(scr_perc = 0.2)
    mw.run(value_object.run(mc.run(init_pos_non_fins,data=d), data=d), data=d)   
    mw.final_positions = bt.limit_pos_size(mw.final_positions)
    ret['value'] = mw.calc_ret(d.basic_data['price_monthly'])

    mw.run(value_object.run(for_object.run(mc.run(init_pos_non_fins,data=d), data=d), data=d), data=d)
    mw.final_positions = bt.limit_pos_size(mw.final_positions)
    ret['for->val'] = mw.calc_ret(d.basic_data['price_monthly']) # still need to limit to 0.2 !!!
    
    mw.run(fs_object.run(value_object.run(for_object.run(mc.run(init_pos_non_fins,data=d), data=d), data=d), data=d), data=d)
    mw.final_positions = bt.limit_pos_size(mw.final_positions)
    ret['f->v->fs'] = mw.calc_ret(d.basic_data['price_monthly']) # still need to limit to 0.2 !!!
    
    mw.run(qual_object.run(value_object.run(for_object.run(mc.run(init_pos_non_fins,data=d), data=d), data=d), data=d), data=d)
    mw.final_positions = bt.limit_pos_size(mw.final_positions)
    ret['QV_test'] = mw.calc_ret(d.basic_data['price_monthly']) # still need to limit to 0.2 !!!
    
    metrics=(['acc','pman','pfd','for','value','pfp','fs','qual','for->val','f->v->fs','QV_test','alsi']) #,'QV_fin'
                                
    bt.plot_returns(ret, metrics)    
    bt.tabulate_results(ret, metrics, frequency=12.0, risk_free = 0.07) # risk free rate is per annum
    bt.plot_CAGR(ret, metrics, 1)
    bt.plot_CAGR(ret, metrics, 5)

# this runs all of the component parts of the QV strategies as separate sub-strategies (for financial companies) 
def QV_fin_parts():
    ret={}  
    mc=bt.Mkt_cap_scr(threshold=0)
    mw = bt.Mkt_cap_weights()
    #mw = bt.Mvi_weights()
    
    #QV_object = bt.QV(threshold=1000, scr2_perc=0.95, scr3_perc = 0.15, scr4_perc = 0.6, upper_limit=0.2)
    #QV_object.backtest(init_pos= init_pos1, data=d) #, rebalance = 'M'                                  
   #ret['QV'] = QV_object.calc_ret(price_data= d.basic_data['price_monthly'])
    
    #QV_fin_object = bt.QV_fin(banks, insurers, threshold=0, scr2_perc = 0.15, scr3_perc = 0.6, upper_limit=0.2)
    #QV_fin_object.backtest(init_pos= init_pos_fin, data=d) #, rebalance = 'M',                                                             
    #ret['QV_fin'] = QV_fin_object.calc_ret(price_data= d.basic_data['price_monthly'])
    
    BAH_object = bt.BAH()
    mw.run(BAH_object.backtest(mc.run(init_pos_fin,data=d), data=d), data=d)
    mw.final_positions = bt.limit_pos_size(mw.final_positions)
    ret['alsi'] = mw.calc_ret(d.basic_data['price_monthly'])  
    
    acc_object = bt.Accruals_fin(scr_perc = 0.05)
    mw.run(acc_object.run(mc.run(init_pos_fin,data=d), data=d), data=d)
    mw.final_positions = bt.limit_pos_size(mw.final_positions)
    ret['acc'] = mw.calc_ret(d.basic_data['price_monthly'])
    
    pman_object = bt.Pman_fin(banks, insurers, scr_perc = 0.05)
    mw.run(pman_object.run(mc.run(init_pos_fin,data=d), data=d), data=d)
    mw.final_positions = bt.limit_pos_size(mw.final_positions)
    ret['pman'] = mw.calc_ret(d.basic_data['price_monthly'])
    
    pfd_object = bt.Pfd_fin(scr_perc = 0.05)
    mw.run(pfd_object.run(mc.run(init_pos_fin,data=d), data=d), data=d)
    mw.final_positions = bt.limit_pos_size(mw.final_positions)
    ret['pfd'] = mw.calc_ret(d.basic_data['price_monthly'])
    
    for_object = bt.Forensic_fin(banks, insurers, scr_perc = 0.05)
    mw.run(for_object.run(mc.run(init_pos_fin,data=d), data=d), data=d)
    mw.final_positions = bt.limit_pos_size(mw.final_positions)
    ret['for'] = mw.calc_ret(d.basic_data['price_monthly'])
    
    fs_object = bt.Fs_fin(banks, insurers, scr_perc = 0.6)
    mw.run(fs_object.run(mc.run(init_pos_fin,data=d), data=d), data=d)
    mw.final_positions = bt.limit_pos_size(mw.final_positions)
    ret['fs'] = mw.calc_ret(d.basic_data['price_monthly'])
    
    pfp_object = bt.Pfp_fin(scr_perc = 0.6)
    mw.run(pfp_object.run(mc.run(init_pos_fin,data=d), data=d), data=d)
    mw.final_positions = bt.limit_pos_size(mw.final_positions)
    ret['pfp'] = mw.calc_ret(d.basic_data['price_monthly'])
    
    qual_object = bt.Quality_fin(banks, insurers, scr_perc = 0.6)
    mw.run(qual_object.run(mc.run(init_pos_fin,data=d), data=d), data=d)
    mw.final_positions = bt.limit_pos_size(mw.final_positions)
    ret['qual'] = mw.calc_ret(d.basic_data['price_monthly'])
    
    value_object = bt.Value_fin(scr_perc = 0.15)
    mw.run(value_object.run(mc.run(init_pos_fin,data=d), data=d), data=d)   
    mw.final_positions = bt.limit_pos_size(mw.final_positions)
    ret['value'] = mw.calc_ret(d.basic_data['price_monthly'])

    mw.run(value_object.run(for_object.run(mc.run(init_pos_fin,data=d), data=d), data=d), data=d)
    mw.final_positions = bt.limit_pos_size(mw.final_positions)
    ret['for->val'] = mw.calc_ret(d.basic_data['price_monthly']) # still need to limit to 0.2 !!!
    
    mw.run(fs_object.run(value_object.run(for_object.run(mc.run(init_pos_fin,data=d), data=d), data=d), data=d), data=d)
    mw.final_positions = bt.limit_pos_size(mw.final_positions)
    ret['f->v->fs'] = mw.calc_ret(d.basic_data['price_monthly']) # still need to limit to 0.2 !!!
    
    mw.run(qual_object.run(value_object.run(for_object.run(mc.run(init_pos_fin,data=d), data=d), data=d), data=d), data=d)
    mw.final_positions = bt.limit_pos_size(mw.final_positions)
    ret['QV_test'] = mw.calc_ret(d.basic_data['price_monthly']) # still need to limit to 0.2 !!!
    
    metrics=(['acc','pman','pfd','for','value','pfp','fs','qual','for->val','f->v->fs','QV_test','alsi']) #,'QV_fin'
                                
    bt.plot_returns(ret, metrics)    
    bt.tabulate_results(ret, metrics, frequency=12.0, risk_free = 0.07) # risk free rate is per annum
    bt.plot_CAGR(ret, metrics, 1)
    bt.plot_CAGR(ret, metrics, 5)

# this runs a random screen strategy, as a baseline comparison (as opposed to just comparing to holding the whole universe)
def random_test():
    ret={}  
    mc=bt.Mkt_cap_scr(threshold=0)
    mw = bt.Mkt_cap_weights()
    #mw = bt.Mvi_weights()
    
    #QV_object = bt.QV(threshold=1000, scr2_perc=0.95, scr3_perc = 0.15, scr4_perc = 0.6, upper_limit=0.2)
    #QV_object.backtest(init_pos= init_pos1, data=d) #, rebalance = 'M'                                  
   #ret['QV'] = QV_object.calc_ret(price_data= d.basic_data['price_monthly'])
    
    #QV_fin_object = bt.QV_fin(banks, insurers, threshold=0, scr2_perc = 0.15, scr3_perc = 0.6, upper_limit=0.2)
    #QV_fin_object.backtest(init_pos= init_pos_fin, data=d) #, rebalance = 'M',                                                             
    #ret['QV_fin'] = QV_fin_object.calc_ret(price_data= d.basic_data['price_monthly'])
    
    BAH_object = bt.BAH()
    mw.run(BAH_object.backtest(mc.run(init_pos_non_fins,data=d), data=d), data=d)
    mw.final_positions = bt.limit_pos_size(mw.final_positions)
    ret['alsi'] = mw.calc_ret(d.basic_data['price_monthly'])  
    
    val_object = bt.Value(scr_perc = 0.15)
    mw.run(val_object.run(mc.run(init_pos_non_fins,data=d), data=d), data=d)
    mw.final_positions = bt.limit_pos_size(mw.final_positions)
    ret['value'] = mw.calc_ret(d.basic_data['price_monthly'])
    
    import numpy as np
    rs_obj = bt.Random_screen(scr_perc = 0.15)
    runs=100    
    test=pd.Series(index=range(runs))
    for i in range(runs):
        mw.run(rs_obj.run(mc.run(init_pos_non_fins,data=d)), data=d)
        mw.final_positions = bt.limit_pos_size(mw.final_positions)
        test[i] = ((np.prod(1.+mw.calc_ret(d.basic_data['price_monthly'])))**(frequency/len(mw.calc_ret(d.basic_data['price_monthly']))))-1
    #random_CAGR = test.mean()
    value_CAGR = ((np.prod(1.+ret['value']))**(frequency/len(ret['value'])))-1
    idx = test  < value_CAGR # how many of the random results are less than our value result (CAGR)
    print "Percentage of random outcomes that are less than our outcome:"
    print "{:.2%}".format(idx.sum()/runs) # what percentage of the random results does our value result outperform?
    # need to calculate what % of the random_CAGR's my value_CAGR beats
    metrics=range(10)
    
    #(['value','random','alsi']) #,'QV_fin'
                                
    bt.plot_returns(ret, metrics)    
    bt.tabulate_results(ret, metrics, frequency=12.0, risk_free = 0.07) # risk free rate is per annum
    
    
    bt.plot_CAGR(ret, metrics, 1)
    bt.plot_CAGR(ret, metrics, 5)
    
def testing_QM():
    ret={}  
    
    QM_object={}
    mom_params=[0.05,0.1, 0.15,0.2,0.25]
    fip_params=[0.55,0.6,0.65,0.7,0.75]
    for i in (mom_params):
        for j in (fip_params):
            QM_object[i,j] = bt.QM(threshold=1000, scr2_perc = i, scr3_perc = j, upper_limit=0.2)
            QM_object[i,j].backtest(init_pos= init_pos1, data = d) #, rebalance = 'M', #Q-NOV, Q-DEC, Q-OCT                        
            ret[i,j] = QM_object[i,j].calc_ret(price_data= d.basic_data['price_monthly'])

    metrics=[(x,y) for x in mom_params for y in fip_params]
                        
    #bt.plot_returns(ret, metrics)    
    bt.tabulate_results(ret, metrics, frequency=12.0, risk_free = 0.07) # risk free rate is per annum
    #bt.plot_CAGR(ret, metrics, 1)
    #bt.plot_CAGR(ret, metrics, 5)  
    plotGPR_3d(mom_params, fip_params)

def testing_QV():
    ret={}  
    
    QV_object={}
    mom_params=[0.05,0.1, 0.15,0.2,0.25]
    fip_params=[0.45,0.5,0.55,0.6,0.65]
    for i in (mom_params):
        for j in (fip_params):
            QV_object[i,j] = bt.QV(threshold=1000, scr2_perc=0.95, scr3_perc = i, scr4_perc = j, upper_limit=0.2)
            QV_object[i,j].backtest(init_pos= init_pos1, data=d) #, rebalance = 'M'                                  
            ret[i,j] = QV_object[i,j].calc_ret(price_data= d.basic_data['price_monthly'])

    metrics=[(x,y) for x in mom_params for y in fip_params]
                        
    #bt.plot_returns(ret, metrics)    
    bt.tabulate_results(ret, metrics, frequency=12.0, risk_free = 0.07) # risk free rate is per annum
    #bt.plot_CAGR(ret, metrics, 1)
    #bt.plot_CAGR(ret, metrics, 5)  
    plotGPR_3d(mom_params, fip_params) 
    
def plotGPR_3d(x_params, y_params):
    import numpy as np
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    from scipy.interpolate import griddata
    
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    # note this: you can skip rows!
    X,Y = np.meshgrid(x_params,y_params)
    X = X.flatten(order='F')
    Y = Y.flatten(order='F') 
    Z = [sum(ret[(x,y)])/abs(sum(ret[(x,y)][ret[(x,y)]<0]))for x in mom_params for y in fip_params] #GPR
    
    xi = np.linspace(min(X),max(X),100)
    yi = np.linspace(min(Y),max(Y),100)
    # VERY IMPORTANT, to tell matplotlib how is your data organized
    zi = griddata((X, Y), Z, (xi[None,:], yi[:,None]), method='cubic')
    
    CS = plt.contour(xi,yi,zi,15,linewidths=0.5,color='k')
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    
    xig, yig = np.meshgrid(xi, yi)
    
    surf = ax.plot_surface(xig, yig, zi,
            linewidth=0)
    
    plt.show()

import numpy as np
### form the 3d random array:
runs = 10
random_df = np.random.rand(runs,init_pos_non_fins.shape[0],init_pos_non_fins.shape[1])    #

### do the market cap screen:
mc=bt.Mkt_cap_scr(threshold=1000)
mc.run(init_pos_non_fins,data=d) 
random_df = np.multiply(random_df,mc.final_positions[init_pos_non_fins.columns].as_matrix()[np.newaxis]) #random_df = random_df*mc.final_positions
#random_df = random_df.replace(0,np.nan)
perc=np.nanpercentile(random_df,80,axis=2,keepdims=True)
screen=np.where(random_df>perc,1,0)

#### do the weighting:
weighting_data = d.basic_data['market_val'][init_pos_non_fins.columns].as_matrix()
weighting_data = (screen*weighting_data)
weights=np.transpose(np.transpose(weighting_data,(0,2,1))/weighting_data.sum(axis=2)[:,np.newaxis],(0,2,1))

## run the limit pos size:
weights = np.where(weights<0.02,0,weights)
init_weights=weights.clip(max=0.2)
row_sum = init_weights.sum(axis=2)
new_weights = np.transpose((np.transpose(init_weights,(0,2,1))/row_sum[:,np.newaxis]),(0,2,1))
new_weights = new_weights.clip(max=0.2)
while new_weights.sum() < 0.9*float(new_weights.shape[0]*new_weights.shape[1]): #while new_weights.sum(axis=1)<1:
        row_sum = new_weights.sum(axis=2)
        new_weights = np.transpose((np.transpose(new_weights,(0,2,1))/row_sum[:,np.newaxis]),(0,2,1))
        new_weights = new_weights.clip(max=0.2)

### calc returns:
start_date = '2000-01-30'
prices = d.basic_data['price_monthly'] 
wp = pd.Panel(new_weights, major_axis=init_pos_non_fins.index, minor_axis=init_pos_non_fins.columns)
pnl = wp.shift(1).mul((prices-prices.shift(1))/prices.shift(1)) # calculate pnl as position yesterday x price change since yesterday
pnl=pnl.fillna(0)
#pnl[positions.shift(1).fillna(0)!=positions.shift(2).fillna(0)]-= cost  #subtract transaction costs:
total_pnl=pnl.sum(axis=2) # sum across all tickers to get total pnl per day
total_positions=wp.sum(axis=2) # sum across tickers to get total number of positions
ret=total_pnl.mul(1/total_positions.shift(1)) # divide pnl by total weight of position in market to get return
ret[ret==-np.inf]=0 # zero out the infs - a problem creeps in because of 27/4/05??
ret=ret.fillna(0).ix[start_date:]

value_object = bt.Value(scr_perc = 0.2)
mw = bt.Mvi_weights()
mw.run(value_object.run(mc.run(init_pos_non_fins,data=d), data=d), data=d)   
mw1=mw.final_positions[mw.final_positions>0].count(axis=1)
mw.final_positions = bt.limit_pos_size(mw.final_positions)
value = mw.calc_ret(d.basic_data['price_monthly'])
#port_size1 = mw.final_positions[mw.final_positions>0].count(axis=1)

#mom_object = bt.Mom(scr_perc = 0.2)
#mw = bt.Mkt_cap_weights()
#mw.run(mom_object.run(mc.run(init_pos1,data=d), data=d), data=d)   
#mw.final_positions = bt.limit_pos_size(mw.final_positions)
#mom = mw.calc_ret(d.basic_data['price_monthly'])
#port_size2 = mw.final_positions[mw.final_positions>0].count(axis=1)

import matplotlib.pyplot as plt
plt.plot(100*np.cumprod(1.+ret))
plt.plot(100*np.cumprod(1.+value),lw=2)
#plt.plot(100*np.cumprod(1.+mom),lw=2)
plt.show()

#plt.plot(mc.final_positions[mc.final_positions>0].count(axis=1))
#plt.plot(mom_object.final_positions[mom_object.final_positions>0].count(axis=1))
#plt.plot(mw.final_positions[mw.final_positions>0].count(axis=1))
#plt.show()
#plt.plot(mw1, label='vmi')
#plt.plot(init_pos_non_fins.count(axis=1), label='init')
#plt.plot(mc.final_positions[mc.final_positions>0].count(axis=1), label='mkt_cap')
def plot_size (scr_perc, mid_object=value_object):
    avail_for_value = mc.final_positions[mc.final_positions>0].count(axis=1)*scr_perc.astype(int)
    plt.plot(avail_for_value, label='avail')
    plt.plot(mid_object.final_positions[mid_object.final_positions>0].count(axis=1), label='chosen')
    plt.plot(mw.final_positions[mw.final_positions>0].count(axis=1), label='limit')
    plt.legend(loc=2)
    plt.show()
    
test=init_pos_non_fins*d.basic_data['market_val']
test2=init_pos_non_fins*d.basic_data['price_monthly']    
#plt.plot(test[test>0].count(axis=1), label='mc_data')
#plt.plot(test2[test2>0].count(axis=1), label='price')
plt.show()


test=pd.DataFrame(index=d.daily_price.index, columns=d.daily_price.columns)
test2=mw.final_positions[d.daily_price.columns]
test.loc[test2.index]=test2
for i in range(1000,len(test)):
    if np.isfinite(test.iloc[i].sum()):
        pass
    else:
        try:
            test.iloc[i]=(test.iloc[i-1]*d.daily_price.pct_change().iloc[i])/(test.iloc[i-1]*d.daily_price.pct_change().iloc[i]).sum()
        except:
            pass

# output the results
daily_returns = bt.calc_ret(test,d.daily_price)
import matplotlib.pyplot as plt
plt.plot(100*np.cumprod(1.+daily_returns))
plt.show()    
    
