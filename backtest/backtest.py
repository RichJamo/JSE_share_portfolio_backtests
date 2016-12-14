# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 10:27:01 2016

@author: Richard
"""
import pandas as pd
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
import os.path
from scipy.stats import norm

############ FUNCTIONS
def max_dd(ser):
    max2here = pd.expanding_max(ser)
    dd2here = (ser - max2here)/max2here
    dd2here[dd2here == -np.inf] = 0
    return dd2here.min()

def data_clean(df): 
    df = df.T.fillna(df.mean(axis=1)).T # replace nans in rows with the mean of that row - only if it isn't a row of nan's
    df = df.interpolate() # interpolate where there are values before and after
    #df = df.fillna(method='bfill') # OPTIONAL: fill backwards where there are no values going back...
    return df
        
def read_in_data(filename):
    df = pd.read_csv(filename)
    try:
        df = df.drop_duplicates()
    except:
        pass
    df = df.dropna(axis=1,how='all') # get rid of columns with all nans
    df = df.dropna(axis=0,how='all') # get rid of rows with all nans
    df = df.set_index('Company')
    df = df.transpose()
    #df = df.fillna(0)
    try:
        df=df.drop(['42671']) # this is for the daily data - weirdly adds in this extra row / column
    except:
        pass
    try:
        df=df.drop(['42675'])
    except:
        pass
    df.index=pd.to_datetime(df.index.astype(str), format='%d/%m/%Y')
    try:
        del df['STP-JSE'] # there's a woopsie in the price data for this stock, so I've just excluded it for now...
    except:
        pass
    try:
        del df['WES-JSE'] # there's a woopsie in the price data for this stock, so I've just excluded it for now...
    except:
        pass
    return df

def screen(init_pos, data_in, scr_perc=0.10, ascending=True):
    tmp_data=data_in*init_pos
    tmp_data=tmp_data.replace(0,np.nan)
    tmp_data=tmp_data.as_matrix()    
    if ascending == True:   
        perc=np.nanpercentile(tmp_data,100*scr_perc,axis=1, keepdims=True)    
        screen=np.where(tmp_data<perc,1,0)
    elif ascending == False:
        perc=np.nanpercentile(tmp_data,100-100*scr_perc,axis=1, keepdims=True)    
        screen=np.where(tmp_data>perc,1,0)
    screen = pd.DataFrame(screen, index=init_pos.index, columns = init_pos.columns)
    return screen



#def random_screen(init_pos, scr_perc=0.10):
#    random_df = pd.DataFrame(np.random.rand(init_pos.shape[0],init_pos.shape[1]), index=init_pos.index, columns = init_pos.columns) #np.random.randint(0,init_pos.shape[1],size=init_pos.shape), index=init_pos.index, columns = init_pos.columns)    
#    tmp_data=random_df*init_pos
#    tmp_data=tmp_data.replace(0,np.nan)
#    data_count = tmp_data.count(axis=1)
#    tmp_screen = tmp_data.rank(axis=1, ascending=True) # change this to generate random numbers
#    data_list = tmp_screen.columns.tolist()
#    for i in data_list:
#        tmp_screen[i][tmp_screen[i]>data_count*scr_perc]=0
#    tmp_screen[tmp_screen>0]=1
#    tmp_screen = tmp_screen.fillna(0)
#    return tmp_screen
    
def bool_screen(init_pos, data_in, threshold): 
    tmp_data=data_in*1    
    tmp_data[tmp_data<threshold]=0
    tmp_data[tmp_data>=threshold]=1
    tmp_screen = tmp_data.fillna(0)
    positions = tmp_screen*init_pos
    return positions
    
def weight(positions, weighting_data):
    positions = ((positions*weighting_data).T/(positions*weighting_data).sum(axis=1)).T
    return positions

def limit_pos_size(weights, lower_limit=0.02, upper_limit = 0.2):     
    weights = weights.clip(upper=upper_limit)
    row_sum = weights.sum(axis=1)
    row_sum_2 = weights[weights<>upper_limit].sum(axis=1)
    new_weights = (weights.T*(1+(1-row_sum)/(row_sum_2))).T
    new_weights = new_weights.clip(upper=upper_limit)
    for j in range(5):
        for i in range(5): #while new_weights.sum(axis=1)<1:
            row_sum = new_weights.sum(axis=1)
            row_sum_2 = new_weights[new_weights<>upper_limit].sum(axis=1)
            new_weights = (new_weights.T*(1+(1-row_sum)/(row_sum_2))).T
            new_weights = new_weights.clip(upper=upper_limit)
        new_weights = new_weights.where(new_weights>lower_limit,0)
    
    return new_weights   
    
def calc_ret(positions, prices, cost=0): #reinsert cost here!!
    pnl = positions.shift(1)*(prices-prices.shift(1))/prices.shift(1) # calculate pnl as position yesterday x price change since yesterday
    pnl=pnl.fillna(0)
    #pnl[positions.shift(1).fillna(0)!=positions.shift(2).fillna(0)]-= cost  #subtract transaction costs:
    total_pnl=pnl.sum(axis=1) # sum across all tickers to get total pnl per day
    total_positions=positions.sum(axis=1) # sum across tickers to get total number of positions
    ret=total_pnl / total_positions.shift(1) # divide pnl by total weight of position in market to get return
    ret[ret==-np.inf]=0 # zero out the infs - a problem creeps in because of 27/4/05??
    ret=ret.fillna(0)
    return ret

def trend_filter(j203_price, ret, risk_free): # all three of these are series
    tmp=pd.concat([j203_price,pd.rolling_mean(j203_price, 12)],axis=1) # create a temp dataframe with alsi price & 12 mth MA
    tmp=tmp.reindex(index=ret.index,method='nearest')
    tmp['diff']=tmp.iloc[:,0]-tmp.iloc[:,1] # diff btw price & MA
    tmp=tmp.shift(1)
    tmp[tmp>0]=0
    tmp[tmp<0]=1 # we generate a signal when last months price below MA
    risk_free=risk_free.reindex(index=tmp.index, method='nearest')
    tmp['returns']=tmp['diff']*risk_free # in months where there's a signal, we take the risk-free rate
    ret[tmp['diff']==1]=tmp['returns'] # we insert the risk-free return into the ret dataframe

def plot_returns(ret, metrics):
    fig, ax1 = plt.subplots()
    colors = ['red','blue','green','magenta','pink','orange', 'purple','yellow','black','cyan','turquoise','white']
    for i in range(len(metrics)):
        plt.plot(100*np.cumprod(1.+ret[metrics[i]]), color=colors[i], lw=1, label = metrics[i])        
    #plt.yscale('log')
    plt.grid(True)
    plt.legend(loc=0)
    ax1.ylabel = ('return')
    plt.xlabel('date')
    ax2 = ax1.twinx()
    #plt.plot(QV_object.final_positions[QV_object.final_positions>0].count(axis=1),'ro') # this is the number of stocks
    #plt.ylabel('no of stocks')
    plt.title('Value of $100 invested (log scale)')   
    plt.show()
    
def plot_CAGR(ret, metrics, num_years):
    rolling_CAGR={}
    colors = ['red','blue','green','magenta','pink','orange', 'purple','yellow','black','cyan','turquoise','white']
    for i in range(len(metrics)):
        rolling_CAGR[metrics[i]] = pd.rolling_apply(ret[metrics[i]], num_years*12, lambda x: (np.prod(1+x)**(12.0/len(x))-1))
        plt.plot(rolling_CAGR[metrics[i]], color=colors[i], lw=1, label = metrics[i])
    plt.grid(True)
    plt.legend(loc=2)
    plt.title('{} yr rolling CAGR'.format(num_years)) 
    plt.show()
    
def tabulate_results(ret, metrics, frequency = 12.0, risk_free = 0.07):  
    MAR=(1+risk_free)**(1.0/frequency)-1 # an annual rate of 7% converted to a monthly rate
    table_list = [[metrics[i], 
                   "{:.2%}".format(((np.prod(1.+ret[metrics[i]]))**(frequency/len(ret[metrics[i]])))-1), #APR
                    "{:.2%}".format(np.sqrt(frequency)*np.std(ret[metrics[i]])),
                    "{:.3}".format(np.sqrt(12)*(ret[metrics[i]].mean()-MAR)/np.std(ret[metrics[i]])), #Sharpe - need to deduce risk-free rate
                    "{:.3}".format(np.sqrt(frequency)*(np.mean(ret[metrics[i]])-MAR)/np.sqrt(sum(((ret[metrics[i]]-MAR)[(ret[metrics[i]]-MAR)<0])**2/len((ret[metrics[i]]-MAR)[(ret[metrics[i]]-MAR)<0])))),
                    "{:.3}".format(sum(ret[metrics[i]])/abs(sum(ret[metrics[i]][ret[metrics[i]]<0]))), #
                    "{:.2%}".format(max_dd(np.cumprod(1.+ret[metrics[i]]))),
                    "{:.2%}".format(max(ret[metrics[i]])),
                    "{:.2%}".format(min(ret[metrics[i]])),
                    "{:.2%}".format(float(ret[metrics[i]][ret[metrics[i]]>0].count())/ret[metrics[i]].count())] # max drawdown - need to make it a percentage
                    for i in range(len(metrics))]
    print tabulate(table_list, headers=['CAGR','Std Dev', 'Sharpe','Sortino','GPR','Max Drawdown','Best mth', 'Worst mth','Win mths']) #,floatfmt=".2%"

def delisting(prices, delist_value):
    prices = prices.replace(0,np.nan)
    columns= prices.columns
    for i in range(len(columns)):
        try:
            row_number = prices.index.get_loc(prices[columns[i]].last_valid_index()) # get ordinal location of the index of the last valid data point in the column
            prices[columns[i]][row_number+1] = prices[columns[i]][row_number]*delist_value
        except:
            pass
    return prices
############## Classes ####################
        
class Data(object):
    
    def __init__(self, path, inputs, inputs_to_shift, months_delay_data=3, start=0, delist_value=0.5):        
        self.months_delay_data = months_delay_data # should this be here, couldn't they be inputs??
        self.start = start # should this be here, couldn't they be inputs??
        self.delist_value = delist_value
        self.inputs = inputs
        self.inputs_to_shift = inputs_to_shift
        self.path = path
        self.basic_data = {}
    
        for i in range(len(self.inputs)):
            self.basic_data[self.inputs[i]] = read_in_data(os.path.join(self.path, self.inputs[i]+'.csv'))
        
        data_to_shift = self.inputs_to_shift
        for i in range(len(data_to_shift)):
            self.basic_data[data_to_shift[i]] = self.basic_data[data_to_shift[i]].shift(months_delay_data)
        
        data_to_clip = ['bps','ebit_oper','entrpr_val', 'ebit_oper']
        for i in range(len(data_to_clip)):
            self.basic_data[data_to_clip[i]] = np.clip(self.basic_data[data_to_clip[i]],1,100000000) # this clipping takes care of negative values
    
        self.index = self.basic_data['price_monthly'].index
        self.index = self.index[start:] # we just go from the date when the valuation data starts...
        for i in range(len(self.inputs)):
            self.basic_data[self.inputs[i]]= self.basic_data[self.inputs[i]].reindex(index=self.index, method='nearest')
            
        ############## HANDLING DELISTINGS
        self.basic_data['price_monthly']= delisting(self.basic_data['price_monthly'],delist_value)
        
        self.daily_price = read_in_data(os.path.join(self.path,'other data','price_daily.csv')).replace(0,np.nan)
        
        self.j203_price = read_in_data(os.path.join(self.path,'other data','j203_price.csv')) # this is monthly
        self.j203_price = self.j203_price.reindex(index=self.index, method='nearest') 
        
        self.risk_free =  pd.read_csv(os.path.join(self.path,'other data','risk_free_rate.csv'))
        self.risk_free = self.risk_free.dropna(axis=1,how='all') # get rid of columns with all nans
        self.risk_free = self.risk_free.dropna(axis=0,how='all') # get rid of rows with all nans
        self.risk_free = self.risk_free.set_index('Unnamed: 0')
        self.risk_free = self.risk_free.transpose()
        self.risk_free.index=pd.to_datetime(self.risk_free.index.astype(str), format='%d/%m/%Y')
        self.risk_free = self.risk_free['TRYZA10Y-FDS']
        self.risk_free = self.risk_free.reindex(index=self.index, method='nearest')
        self.risk_free = self.risk_free.fillna(method='pad')
        self.risk_free=self.risk_free.astype(float)
        self.risk_free = self.risk_free-2
        self.risk_free=self.risk_free/100
        self.risk_free = self.risk_free.apply(lambda x: (x+1)**(1.0/12)-1)

class Strategy(object):
    
    def liquidity(self, data):
        liquidity = data.basic_data['volume_monthly']/data.basic_data['free_float']
        return liquidity
    
    def market_cap(self, data):
        return data.basic_data['market_val']
    
    def equal_weight(self, data):
        equal_weight = pd.DataFrame(1, index=data.basic_data['market_val'].index, columns=data.basic_data['market_val'].columns)
        return equal_weight
         
    def mkt_weight(self, data):
        mkt_weight = data.basic_data['market_val']
        return mkt_weight
        
    def mvi_weight(self, data, mvi_window_len=220):
        stdev = pd.rolling_std(data.daily_price, window=mvi_window_len)
        mvi_weight = 1/stdev #daily_price/stdev - this is the volatility indicator
        #test=daily_price.diff()
        #downside_dev = pd.rolling_apply(test, 110, lambda x: np.sqrt((x[x<0]-x.mean())**2).sum()/len(x[x<0]) )
        mvi_weight= mvi_weight.reindex(index=data.index, method='nearest')    
        return mvi_weight
        
    def set_weights(self,data, mvi_window_len=220):
        weighting = {}
        weighting['equal'] = pd.DataFrame(1, index=data.basic_data['market_val'].index, columns=data.basic_data['market_val'].columns)
        weighting['mkt_cap'] = data.basic_data['market_val']
        stdev = pd.rolling_std(data.daily_price, window=mvi_window_len)
        mvi_weight = 1/stdev #daily_price/stdev - this is the volatility indicator
        #test=daily_price.diff()
        #downside_dev = pd.rolling_apply(test, 110, lambda x: np.sqrt((x[x<0]-x.mean())**2).sum()/len(x[x<0]) )
        weighting['mvi'] = mvi_weight.reindex(index=data.index, method='nearest')
        return weighting
        
    def calc_ret(self, price_data, start_date='2000-01-30'):
        ret = calc_ret(self.final_positions.ix[start_date:], price_data.ix[start_date:])
        return ret
        
    def latest(self, date='2016-11-01'):
        latest = self.final_positions.ix[date][self.final_positions.ix[date]>0]
        return latest  
        
class QV(Strategy):
    
    def __init__(self, threshold=1000, scr2_perc=0.95, scr3_perc = 0.15, scr4_perc = 0.6, weighting='mkt_cap', upper_limit=0.2): #, trend_filter=False, threshold = 0
        #self.rebalance = rebalance
        self.threshold = threshold
        self.scr2_perc = scr2_perc# forensics -> positions 2
        self.scr3_perc = scr3_perc # value -> positions 3
        self.scr4_perc = scr4_perc # quality -> positions 4
        self.upper_limit = upper_limit
        self.weighting = weighting
        
    def backtest(self, init_pos, data):
        
        self.positions1 = bool_screen(init_pos, self.market_cap(data), threshold=self.threshold) # mkt cap of R2bn or more
        
        accruals_object = Accruals(scr_perc = self.scr2_perc)
        self.positions2a = accruals_object.run(self.positions1,data)
        
        pman_object = Pman(scr_perc = self.scr2_perc)
        self.positions2b = pman_object.run(self.positions1,data)
        
        pfd_object = Pfd(scr_perc = self.scr2_perc)
        self.positions2c = pfd_object.run(self.positions1,data)
        
        self.positions2 = self.positions2a*self.positions2b*self.positions2c # combine the results of pfd & accrualspositions2 = screen(positions1, data_for_scr2, scr_perc = scr2_perc, ascending=scr2_asc)
        
        value_object = Value(scr_perc = self.scr3_perc)
        self.positions3 = value_object.run(self.positions2, data)
        
        quality_object = Quality(scr_perc=self.scr4_perc)
        self.positions4 = quality_object.run(self.positions3, data)
        
        #mw_object=mkt_cap_weights()
        mw_object=Mvi_weights()
        self.positions5 = mw_object.run(self.positions4,data) #weight(self.positions4, self.set_weights(data)[self.weighting])

        self.final_positions = limit_pos_size(self.positions5, self.upper_limit) 
        
class Accruals(Strategy):
    def __init__(self, scr_perc = 0.05):
        self.scr_perc=scr_perc 
        
    def run(self, init_pos, data): 
        accruals = (self.p_sta(data)+self.p_snoa(data))/2
        positions = screen(init_pos, accruals, scr_perc = self.scr_perc, ascending=True)
        positions = positions.replace(0,np.nan)
        positions = positions.replace(1,0)
        positions = positions.replace(np.nan,1)
        self.final_positions = positions*init_pos        
        ## how do I invert the selection?
        return self.final_positions
        
    def p_sta(self, data):
        sta = abs(data.basic_data['net_inc']-data.basic_data['oper_cf'])/data.basic_data['total assets']
        sta = data_clean(sta)
        p_sta = (sta.rank(axis=1, ascending = True).T/sta.count(axis=1)).T 
        return p_sta
        
    def p_snoa(self, data):
        operating_assets = data.basic_data['total assets']-data.basic_data['cash_st']
        operating_liabs = data.basic_data['total assets']-data.basic_data['debt_st']-data.basic_data['debt_lt']-data.basic_data['pfd_stk']-data.basic_data['min_int_exp']-data.basic_data['com_eq']
        snoa = (operating_assets - operating_liabs)/data.basic_data['total assets']
        snoa = (snoa.T.where(snoa.T>0, -snoa.T+np.max(snoa.T,axis=0))).T #windsorize
        #snoa = data_clean(snoa)
        p_snoa = (snoa.rank(axis=1, ascending = True).T/snoa.count(axis=1)).T
        return p_snoa 
 
class Accruals_fin(Accruals):
        
    def p_snoa(self, data):
        cash = data.basic_data['cash_st']
        cash = cash.where(np.isfinite(cash),data.basic_data['cash_st'])
        cash = cash.where(np.isfinite(cash),data.basic_data['cash_due'])
        
        operating_assets = data.basic_data['total assets']-cash
        
        st_debt = data.basic_data['debt_st']
        st_debt = st_debt.where(np.isfinite(st_debt),data.basic_data['deposits'])
        
        operating_liabs = data.basic_data['total assets']-st_debt-data.basic_data['debt_lt']-data.basic_data['pfd_stk']-data.basic_data['min_int_exp']-data.basic_data['com_eq'] #change std to deposits for banks??
        
        snoa = (operating_assets - operating_liabs)/data.basic_data['total assets']
        snoa = (snoa.T.where(snoa.T>0, -snoa.T+np.max(snoa.T,axis=0))).T #windsorize
        #snoa = data_clean(snoa)
        p_snoa = (snoa.rank(axis=1, ascending = True).T/snoa.count(axis=1)).T
        return p_snoa
        
class Pman(Strategy):
    def __init__(self, scr_perc = 0.05):
        self.scr_perc=scr_perc 
        
    def run(self, init_pos, data): 
        dsri =  data.basic_data['receiv_turn_days'] / data.basic_data['receiv_turn_days'].shift(12)
        #dsri = data_clean(dsri)
        gmi =  data.basic_data['gross_mgn'].shift(12)/ data.basic_data['gross_mgn'] 
        #gmi = data_clean(gmi)
        aqi = (data.basic_data['total assets'] - data.basic_data['assets_curr'] - data.basic_data['ppe_net'] )/ data.basic_data['total assets'] 
        #aqi = data_clean(aqi)
        sgi =  data.basic_data['sales'] / data.basic_data['sales'].shift(12)
        #sgi = data_clean(sgi)
        depi =  data.basic_data['dep_amort'].shift(12)/ data.basic_data['dep_amort'] 
        #depi = data_clean(depi)
        sgai =  data.basic_data['sga'] / data.basic_data['sga'].shift(12)
        #sgai = data_clean (sgai)
        tata = (data.basic_data['wkcap_chg'] - data.basic_data['dep_amort'] )/ data.basic_data['total assets'] 
        #tata = data_clean(tata)
        lvgi =  data.basic_data['debt_assets'] / data.basic_data['debt_assets'].shift(12)
        #lvgi = data_clean(lvgi)
    
        probm = -4.84+0.92*dsri+0.528*gmi+0.404*aqi+0.892*sgi+0.115*depi+4.679*tata-0.172*sgai-0.372*lvgi #
        pman = pd.DataFrame(norm.cdf(probm), index=data.index, columns = data.basic_data['price_monthly'].columns) # tested this, works fine
        
        positions = screen(init_pos, pman, scr_perc = self.scr_perc, ascending=False)
        positions = positions.replace(0,np.nan)
        positions = positions.replace(1,0)
        positions = positions.replace(np.nan,1)
        self.final_positions = positions*init_pos
        return self.final_positions

class Pman_fin(Pman):
    def __init__(self, banks, insurers, scr_perc = 0.05):
        self.scr_perc=scr_perc 
        self.banks = banks
        self.insurers = insurers
        
    def run(self, init_pos, data): 
        dsri =  data.basic_data['receiv_turn_days'] / data.basic_data['receiv_turn_days'] .shift(12) #this is what holding companies will get 
        dsri[self.banks]= data.basic_data['tier1_capital'][self.banks]/data.basic_data['tier1_capital'][self.banks].shift(12) 
        dsri[self.insurers]=data.basic_data['sales_per_emp'][self.insurers]/data.basic_data['sales_per_emp'][self.insurers].shift(12)
        dsri = data_clean(dsri)
        
        gmi =  data.basic_data['gross_mgn'].shift(12)/ data.basic_data['gross_mgn'] #this is what holding companies will get
        gmi[self.banks]= data.basic_data['int_mgn'][self.banks].shift(12)/data.basic_data['int_mgn'][self.banks] 
        gmi[self.insurers]=data.basic_data['oper_mgn'][self.insurers].shift(12)/data.basic_data['oper_mgn'][self.insurers]
        gmi = data_clean(gmi)
        
        aqi = (data.basic_data['total assets'] - data.basic_data['assets_curr'] - data.basic_data['ppe_net'] )/ data.basic_data['total assets'] 
        aqi = data_clean(aqi)
        sgi =  data.basic_data['sales'] / data.basic_data['sales'] .shift(12)
        sgi = data_clean(sgi)
        depi =  data.basic_data['dep_amort'] .shift(12)/ data.basic_data['dep_amort'] 
        depi = data_clean(depi)
        sgai =  data.basic_data['sga'] / data.basic_data['sga'] .shift(12)
        sgai = data_clean (sgai)
        tata = (data.basic_data['wkcap_chg'] - data.basic_data['dep_amort'] )/ data.basic_data['total assets'] 
        tata = data_clean(tata)
        lvgi =  data.basic_data['debt_assets'] / data.basic_data['debt_assets'].shift(12)
        lvgi = data_clean(lvgi)

        probm = -4.84+0.92*dsri+0.528*gmi+0.404*aqi+0.892*sgi+0.115*depi+4.679*tata-0.172*sgai-0.372*lvgi #
        pman = pd.DataFrame(norm.cdf(probm), index=data.index, columns = data.basic_data['price_monthly'].columns)
        
        positions = screen(init_pos, pman, scr_perc = self.scr_perc, ascending=False)
        positions = positions.replace(0,np.nan)
        positions = positions.replace(1,0)
        positions = positions.replace(np.nan,1)
        self.final_positions = positions*init_pos
        return self.final_positions
        
class Pfd(Strategy): 
    def __init__(self, scr_perc = 0.05):
        self.scr_perc=scr_perc 
        
    def run(self, init_pos, data): 
        nimta = data.basic_data['net_inc'] /(data.basic_data['market_val']  + data.basic_data['liabs'] )
        nimta_avg = (0.5333*nimta)+(0.2666*nimta.shift(3))+(nimta.shift(6)*0.1333)+(nimta.shift(9)*0.0666)
        #nimta_avg = data_clean(nimta_avg)
        tlmta = data.basic_data['liabs'] /(data.basic_data['market_val']  + data.basic_data['liabs'] )
        #tlmta = data_clean(tlmta)
        exret = np.add(np.log(1+data.basic_data['price_monthly'] .pct_change(periods=3)),-np.log(1+data.j203_price.pct_change(periods=3))) #will have to get alsi?
        exret_avg = (0.5333*exret)+(0.2666*exret.shift(3))+(exret.shift(6)*0.1333)+(exret.shift(9)*0.0666)
        #exret_avg = data_clean(exret_avg)
        sigma = (np.sqrt(252)*pd.rolling_std(data.daily_price.pct_change(), window=60)).reindex(index=data.index,method='nearest') # is this the right way to annualise??
        #sigma = data_clean(sigma)
        rsize = np.log((data.basic_data['market_val'].T/data.basic_data['market_val'] .sum(axis=1)).T) # log base 10 or natural log???
        #rsize= data_clean(rsize)        
        cashmta = data.basic_data['cash_st'] /(data.basic_data['market_val']  + data.basic_data['liabs'] )
        #cashmta = data_clean(cashmta)
        mb = (data.basic_data['market_val']  + data.basic_data['liabs'] )/ (data.basic_data['total assets'] -data.basic_data['liabs']  + data.basic_data['market_val'] *0.1) # use bps or use total assets - liabs?
        #mb = data_clean(mb)        
        price = np.log(np.clip(data.basic_data['price_monthly'] , a_min = 0, a_max=200))
        #price = data_clean(price)        
        
        lpfd = -20.26*nimta_avg+1.42*tlmta-7.13*exret_avg+1.41*sigma-0.045*rsize-2.13*cashmta+0.075*mb-0.058*price-9.16
        pfd = 1 / (1+np.exp(-lpfd))        
        positions = screen(init_pos, pfd, scr_perc = self.scr_perc, ascending=False)
        positions = positions.replace(0,np.nan)
        positions = positions.replace(1,0)
        positions = positions.replace(np.nan,1)
        self.final_positions = positions*init_pos
        return self.final_positions

class Pfd_fin(Pfd): 
    def __init__(self, scr_perc = 0.05):
        self.scr_perc=scr_perc 
        
    def run(self, init_pos, data): 
        nimta = data.basic_data['net_inc'] /(data.basic_data['market_val']  + data.basic_data['liabs'] )
        nimta_avg = (0.5333*nimta)+(0.2666*nimta.shift(3))+(nimta.shift(6)*0.1333)+(nimta.shift(9)*0.0666)
        nimta_avg = data_clean(nimta_avg)
        tlmta = data.basic_data['liabs'] /(data.basic_data['market_val']  + data.basic_data['liabs'] )
        tlmta = data_clean(tlmta)
        exret = np.add(np.log(1+data.basic_data['price_monthly'] .pct_change(periods=3)),-np.log(1+data.j203_price.pct_change(periods=3))) #will have to get alsi?
        exret_avg = (0.5333*exret)+(0.2666*exret.shift(3))+(exret.shift(6)*0.1333)+(exret.shift(9)*0.0666)
        exret_avg = data_clean(exret_avg)
        sigma = (np.sqrt(252)*pd.rolling_std(data.daily_price.pct_change(), window=60)).reindex(index=data.index,method='nearest') # is this the right way to annualise??
        sigma = data_clean(sigma)
        rsize = np.log((data.basic_data['market_val'] .T/data.basic_data['market_val'] .sum(axis=1)).T) # log base 10 or natural log???
        rsize= data_clean(rsize)        
        
        cashmta = data.basic_data['cash_st'] /(data.basic_data['market_val']  + data.basic_data['liabs'] ) # change this - use either cash_st, cash due from bk, or cash only
        cashmta = cashmta.where(np.isfinite(cashmta), data.basic_data['cash_due'])
        cashmta = cashmta.where(np.isfinite(cashmta), data.basic_data['cash_only'])
        cashmta = data_clean(cashmta)
        
        mb = (data.basic_data['market_val']  + data.basic_data['liabs'] )/ (data.basic_data['total assets'] -data.basic_data['liabs']  + data.basic_data['market_val'] *0.1) # use bps or use total assets - liabs?
        mb = data_clean(mb)        
        price = np.log(np.clip(data.basic_data['price_monthly'] , a_min = 0, a_max=200))
        price = data_clean(price)        
        
        lpfd = -20.26*nimta_avg+1.42*tlmta-7.13*exret_avg+1.41*sigma-0.045*rsize-2.13*cashmta+0.075*mb-0.058*price-9.16
        pfd = 1 / (1+np.exp(-lpfd))       
        positions = screen(init_pos, pfd, scr_perc = self.scr_perc, ascending=False)
        positions = positions.replace(0,np.nan)
        positions = positions.replace(1,0)
        positions = positions.replace(np.nan,1)
        self.final_positions = positions*init_pos
        return self.final_positions
        
class Forensic(Strategy):
    def __init__(self, scr_perc = 0.05):
        self.scr_perc=scr_perc 
        
    def run(self, init_pos, data): 
        accruals_obj = Accruals(scr_perc = self.scr_perc)
        pfd_obj = Pfd(scr_perc = self.scr_perc)
        pman_obj = Pman(scr_perc = self.scr_perc)     
        self.final_positions = accruals_obj.run(init_pos, data) * pfd_obj.run(init_pos, data) * pman_obj.run(init_pos, data)
        return self.final_positions

class Forensic_fin(Forensic):
    def __init__(self, banks, insurers, scr_perc = 0.05):
        self.scr_perc=scr_perc 
        self.banks = banks
        self.insurers = insurers
        
    def run(self, init_pos, data): 
        accruals_obj = Accruals_fin(scr_perc = self.scr_perc)
        pfd_obj = Pfd_fin(scr_perc = self.scr_perc)
        pman_obj = Pman_fin(banks = self.banks, insurers = self.insurers, scr_perc = self.scr_perc)     
        self.final_positions = accruals_obj.run(init_pos, data) * pfd_obj.run(init_pos, data) * pman_obj.run(init_pos, data)
        return self.final_positions
        
class Value(Strategy):
    def __init__(self, scr_perc = 0.15):
        self.scr_perc=scr_perc 
        
    def run(self, init_pos, data): 
        value = data.basic_data['entrpr_val']/data.basic_data['ebit_oper'] # how much it costs / how much it makes - so a a high score = expensive, a low score = cheap
        self.final_positions = screen(init_pos, value, scr_perc = self.scr_perc, ascending=True)
        return self.final_positions

#    def value2(self, data):
#        value = data.basic_data['price_monthly']/data.basic_data['bps'] # how much it costs / how much it makes - so a a high score = expensive, a low score = cheap
#        #value = data_clean(value)
#        return value
class Value_fin(Value):
    def __init__(self, scr_perc = 0.15):
        self.scr_perc=scr_perc 
        
    def run(self, init_pos, data): 
        value = data.basic_data['entrpr_val']/data.basic_data['ebit_oper'] # should be ebit_oper??
        value = value.where(np.isfinite(value), data.basic_data['entrpr_val']/data.basic_data['oper_inc']) # should be oper_inc?
        self.final_positions = screen(init_pos, value, scr_perc = self.scr_perc, ascending=True)
        return self.final_positions
        
class Quality(Strategy):
    def __init__(self, scr_perc = 0.5):
        self.scr_perc=scr_perc 
        
    def run(self, init_pos, data): 
        pfp_object = Pfp(scr_perc = self.scr_perc)
        fs_object = Fs(scr_perc = self.scr_perc)
        quality = (fs_object.metric(data)+pfp_object.metric(data))/2.0
        self.final_positions = screen(init_pos, quality, scr_perc = self.scr_perc, ascending=False)
        return self.final_positions

class Quality_fin(Quality):
    def __init__(self, banks, insurers, scr_perc = 0.5):
        self.scr_perc=scr_perc 
        self.banks = banks
        self.insurers = insurers
        
    def run(self, init_pos, data): 
        pfp_object = Pfp_fin(scr_perc = self.scr_perc)
        fs_object = Fs_fin(scr_perc = self.scr_perc, banks = self.banks, insurers = self.insurers)
        quality = (fs_object.metric(data)+pfp_object.metric(data))/2.0
        self.final_positions = screen(init_pos, quality, scr_perc = self.scr_perc, ascending=False)
        return self.final_positions
        
class Pfp(Strategy):
    def __init__(self, scr_perc = 0.5):
        self.scr_perc=scr_perc 
        
    def run(self, init_pos, data): 
        pfp = self.metric(data)
        self.final_positions = screen(init_pos, pfp, scr_perc = self.scr_perc, ascending=False)
        return self.final_positions
    
    def metric(self, data):
        mg = ((1+data.basic_data['gross_mgn']/100)*(1+data.basic_data['gross_mgn'].shift(1*12)/100)*(1+data.basic_data['gross_mgn'].shift(2*12)/100)*(1+data.basic_data['gross_mgn'].shift(3*12)/100)*(1+data.basic_data['gross_mgn'].shift(4*12)/100)*(1+data.basic_data['gross_mgn'].shift(5*12)/100)*(1+data.basic_data['gross_mgn'].shift(6*12)/100)*(1+data.basic_data['gross_mgn'].shift(7*12)/100))**(1.0/8)-1 #pd.rolling_apply(data.basic_data['gross_mgn'], 96,lambda x: gmean(1+x/100)-1) #
        #mg = data_clean(mg)
        p_mg = (mg.rank(axis=1, ascending = True).T/mg.count(axis=1)).T         
        #gm_avg= (data.basic_data['gross_mgn']+data.basic_data['gross_mgn'].shift(1*12)+data.basic_data['gross_mgn'].shift(2*12)+data.basic_data['gross_mgn'].shift(3*12))/4.0
        gm_avg= (data.basic_data['gross_mgn']+data.basic_data['gross_mgn'].shift(1*12)+data.basic_data['gross_mgn'].shift(2*12)+data.basic_data['gross_mgn'].shift(3*12)+data.basic_data['gross_mgn'].shift(4*12)+data.basic_data['gross_mgn'].shift(5*12)+data.basic_data['gross_mgn'].shift(6*12)+data.basic_data['gross_mgn'].shift(7*12))/8.0
        #ms = gm_avg/np.sqrt(((data.basic_data['gross_mgn']-gm_avg)**2+(data.basic_data['gross_mgn'].shift(1*12)-gm_avg)**2+(data.basic_data['gross_mgn'].shift(2*12)-gm_avg)**2+(data.basic_data['gross_mgn'].shift(3*12)-gm_avg)**2)/4.0) #pd.rolling_std(data.basic_data['gross_mgn'], fp_len)) #.asfreq('A','pad'),8
        ms = gm_avg/np.sqrt(((data.basic_data['gross_mgn']-gm_avg)**2+(data.basic_data['gross_mgn'].shift(1*12)-gm_avg)**2+(data.basic_data['gross_mgn'].shift(2*12)-gm_avg)**2+(data.basic_data['gross_mgn'].shift(3*12)-gm_avg)**2+(data.basic_data['gross_mgn'].shift(4*12)-gm_avg)**2+(data.basic_data['gross_mgn'].shift(5*12)-gm_avg)**2+(data.basic_data['gross_mgn'].shift(6*12)-gm_avg)**2+(data.basic_data['gross_mgn'].shift(7*12)-gm_avg)**2)/8.0) #pd.rolling_std(data.basic_data['gross_mgn'], fp_len)) #.asfreq('A','pad'),8
        #ms = data_clean(ms)
        p_ms = (ms.rank(axis=1, ascending = True).T/ms.count(axis=1)).T
        mm = pd.DataFrame(np.where(p_mg>p_ms,p_mg,p_ms), index=p_mg.index, columns=p_mg.columns) # or does it have to be max of the percentiles?
        #roa = ((1+data.basic_data['roa']/100)*(1+data.basic_data['roa'].shift(1*12)/100)*(1+data.basic_data['roa'].shift(2*12)/100)*(1+data.basic_data['roa'].shift(3*12)/100))**(0.25)-1 #pd.rolling_apply(data.basic_data['net_inc'].asfreq('A','pad')/data.basic_data['total assets'].asfreq('A','pad'), 8, lambda x: gmean(1+x/100)-1)
        roa = ((1+data.basic_data['roa']/100)*(1+data.basic_data['roa'].shift(1*12)/100)*(1+data.basic_data['roa'].shift(2*12)/100)*(1+data.basic_data['roa'].shift(3*12)/100)*(1+data.basic_data['roa'].shift(4*12)/100)*(1+data.basic_data['roa'].shift(5*12)/100)*(1+data.basic_data['roa'].shift(6*12)/100)*(1+data.basic_data['roa'].shift(7*12)/100))**(1.0/8)-1 #pd.rolling_apply(data.basic_data['net_inc'].asfreq('A','pad')/data.basic_data['total assets'].asfreq('A','pad'), 8, lambda x: gmean(1+x/100)-1)
        #roa = data_clean(roa)
        p_roa = (roa.rank(axis=1, ascending = True).T/roa.count(axis=1)).T
        #roce = ((1+data.basic_data['roic']/100)*(1+data.basic_data['roic'].shift(1*12)/100)*(1+data.basic_data['roic'].shift(2*12)/100)*(1+data.basic_data['roic'].shift(3*12)/100))**(0.25)-1 #pd.rolling_apply(data.basic_data['ebit_oper'].asfreq('A','pad')/data.basic_data['market_val'].asfreq('A','pad'), 8, lambda x: gmean(1+x/100)-1)
        roce = ((1+data.basic_data['roic']/100)*(1+data.basic_data['roic'].shift(1*12)/100)*(1+data.basic_data['roic'].shift(2*12)/100)*(1+data.basic_data['roic'].shift(3*12)/100)*(1+data.basic_data['roic'].shift(4*12)/100)*(1+data.basic_data['roic'].shift(5*12)/100)*(1+data.basic_data['roic'].shift(6*12)/100)*(1+data.basic_data['roic'].shift(7*12)/100))**(1.0/8)-1 #pd.rolling_apply(data.basic_data['ebit_oper'].asfreq('A','pad')/data.basic_data['market_val'].asfreq('A','pad'), 8, lambda x: gmean(1+x/100)-1)
        #roce = data_clean(roce)
        p_roce = (roce.rank(axis=1, ascending = True).T/roce.count(axis=1)).T
        fcfa = (data.basic_data['free_cf']+data.basic_data['free_cf'].shift(1*12)+data.basic_data['free_cf'].shift(2*12)+data.basic_data['free_cf'].shift(3*12)+data.basic_data['free_cf'].shift(4*12)+data.basic_data['free_cf'].shift(5*12)+data.basic_data['free_cf'].shift(6*12)+data.basic_data['free_cf'].shift(7*12))/data.basic_data['total assets']
        #fcfa = data_clean(fcfa)
        p_fcfa = (fcfa.rank(axis=1, ascending = True).T/fcfa.count(axis=1)).T
        pfp = (mm+p_roa+p_roce+p_fcfa)/4.0
        #pfp = pfp.asfreq('A','pad')
        pfp = pfp.reindex(index=data.index, method='nearest')
        #pfp = data_clean(pfp)
        return pfp

class Pfp_fin(Pfp):

    def metric(self, data):
        mg = ((1+data.basic_data['ptx_mgn']/100)*(1+data.basic_data['ptx_mgn'].shift(1*12)/100)*(1+data.basic_data['ptx_mgn'].shift(2*12)/100)*(1+data.basic_data['ptx_mgn'].shift(3*12)/100))**(0.25)-1 #pd.rolling_apply(data.basic_data['gross_mgn'], 96,lambda x: gmean(1+x/100)-1) #
        mg = data_clean(mg)
        p_mg = (mg.rank(axis=1, ascending = True).T/mg.count(axis=1)).T         
        gm_avg= (data.basic_data['ptx_mgn']+data.basic_data['ptx_mgn'].shift(1*12)+data.basic_data['ptx_mgn'].shift(2*12)+data.basic_data['ptx_mgn'].shift(3*12))/4.0
        ms = gm_avg/np.sqrt(((data.basic_data['ptx_mgn']-gm_avg)**2+(data.basic_data['ptx_mgn'].shift(1*12)-gm_avg)**2+(data.basic_data['ptx_mgn'].shift(2*12)-gm_avg)**2+(data.basic_data['ptx_mgn'].shift(3*12)-gm_avg)**2)/4.0) #pd.rolling_std(data.basic_data['gross_mgn'], fp_len)) #.asfreq('A','pad'),8
        ms = data_clean(ms)
        p_ms = (ms.rank(axis=1, ascending = True).T/ms.count(axis=1)).T
        mm = pd.DataFrame(np.where(p_mg>p_ms,p_mg,p_ms), index=p_mg.index, columns=p_mg.columns) # or does it have to be max of the percentiles?
        roa = ((1+data.basic_data['roa']/100)*(1+data.basic_data['roa'].shift(1*12)/100)*(1+data.basic_data['roa'].shift(2*12)/100)*(1+data.basic_data['roa'].shift(3*12)/100))**(0.25)-1 #pd.rolling_apply(data.basic_data['net_inc'].asfreq('A','pad')/data.basic_data['total assets'].asfreq('A','pad'), 8, lambda x: gmean(1+x/100)-1)
        roa = data_clean(roa)
        p_roa = (roa.rank(axis=1, ascending = True).T/roa.count(axis=1)).T
        roce = ((1+data.basic_data['roic']/100)*(1+data.basic_data['roic'].shift(1*12)/100)*(1+data.basic_data['roic'].shift(2*12)/100)*(1+data.basic_data['roic'].shift(3*12)/100))**(0.25)-1 #pd.rolling_apply(data.basic_data['ebit_oper'].asfreq('A','pad')/data.basic_data['market_val'].asfreq('A','pad'), 8, lambda x: gmean(1+x/100)-1)
        roce = data_clean(roce)
        p_roce = (roce.rank(axis=1, ascending = True).T/roce.count(axis=1)).T
        fcfa = (data.basic_data['free_cf']+data.basic_data['free_cf'].shift(1*12)+data.basic_data['free_cf'].shift(2*12)+data.basic_data['free_cf'].shift(3*12))/data.basic_data['total assets']
        fcfa = data_clean(fcfa)
        p_fcfa = (fcfa.rank(axis=1, ascending = True).T/fcfa.count(axis=1)).T
        pfp = (mm+p_roa+p_roce+p_fcfa)/4.0
        #pfp = pfp.asfreq('A','pad')
        pfp = pfp.reindex(index=data.index, method='nearest')
        #pfp = data_clean(pfp)
        return pfp
        
class Fs(Strategy):
    def __init__(self, scr_perc = 0.5):
        self.scr_perc=scr_perc 
        
    def run(self, init_pos, data): 
        fs = self.metric(data)
        self.final_positions = screen(init_pos, fs, scr_perc = self.scr_perc, ascending=False)
        return self.final_positions
    
    def metric(self, data):
        roa = pd.DataFrame(np.where(data.basic_data['roa']>0,1,0),index=data.basic_data['roa'].index, columns=data.basic_data['roa'].columns)
        fcfta = pd.DataFrame(np.where(data.basic_data['free_cf']>0,1,0),index=data.basic_data['free_cf'].index, columns=data.basic_data['free_cf'].columns)
        accrual = pd.DataFrame(np.where(data.basic_data['free_cf']-data.basic_data['roa']>0,1,0),index=data.basic_data['free_cf'].index, columns=data.basic_data['free_cf'].columns)
        chg_lever = pd.DataFrame(np.where(data.basic_data['debt_lt']/data.basic_data['total assets']<data.basic_data['debt_lt'].shift(12)/data.basic_data['total assets'].shift(12),1,0),index=data.basic_data['debt_lt'].index, columns=data.basic_data['debt_lt'].columns)
        chg_liquid = pd.DataFrame(np.where(data.basic_data['curr_ratio']>data.basic_data['curr_ratio'].shift(12),1,0),index=data.basic_data['curr_ratio'].index, columns=data.basic_data['curr_ratio'].columns)
        chg_neqiss = pd.DataFrame(np.where(data.basic_data['com_shs_out']-data.basic_data['com_shs_out'].shift(12)>1,1,0),index=data.basic_data['com_shs_out'].index, columns=data.basic_data['com_shs_out'].columns)
        chg_roa = pd.DataFrame(np.where(data.basic_data['roa']>data.basic_data['roa'].shift(12),1,0),index=data.basic_data['roa'].index, columns=data.basic_data['roa'].columns)
        chg_fcfta = pd.DataFrame(np.where(data.basic_data['free_cf']>data.basic_data['free_cf'].shift(12),1,0),index=data.basic_data['free_cf'].index, columns=data.basic_data['free_cf'].columns)
        chg_margin = pd.DataFrame(np.where(data.basic_data['gross_mgn']>data.basic_data['gross_mgn'].shift(12),1,0),index=data.basic_data['gross_mgn'].index, columns=data.basic_data['gross_mgn'].columns)
        chg_turn = pd.DataFrame(np.where(data.basic_data['asset_turn']>data.basic_data['asset_turn'].shift(12),1,0),index=data.basic_data['asset_turn'].index, columns=data.basic_data['asset_turn'].columns)
        
        p_fs = (roa+fcfta+accrual+chg_lever+chg_roa+chg_liquid+chg_neqiss+chg_fcfta+chg_margin+chg_turn)/10.0
        #### adding this line below to convert this to a percentile, even though they don't do this in the book!?        
        p_fs = (p_fs.rank(axis=1, ascending = True).T/p_fs.count(axis=1)).T #
        return p_fs

class Fs_fin(Fs):
    def __init__(self, banks, insurers, scr_perc = 0.5):
        self.scr_perc=scr_perc 
        self.banks = banks
        self.insurers = insurers
    
    def metric(self, data):
        roa = pd.DataFrame(np.where(data.basic_data['roa']>0,1,0),index=data.basic_data['roa'].index, columns=data.basic_data['roa'].columns)
        fcfta = pd.DataFrame(np.where(data.basic_data['free_cf']>0,1,0),index=data.basic_data['free_cf'].index, columns=data.basic_data['free_cf'].columns)
        accrual = pd.DataFrame(np.where(data.basic_data['free_cf']-data.basic_data['roa']>0,1,0),index=data.basic_data['free_cf'].index, columns=data.basic_data['free_cf'].columns)
        chg_lever = pd.DataFrame(np.where(data.basic_data['debt_lt']/data.basic_data['total assets']<data.basic_data['debt_lt'].shift(12)/data.basic_data['total assets'].shift(12),1,0),index=data.basic_data['debt_lt'].index, columns=data.basic_data['debt_lt'].columns)
        
        ######## should use 'liabs_oth' in this first one, instead of liabs:
        chg_liquid = pd.DataFrame(np.where(data.basic_data['cash_st']/data.basic_data['liab_oth']>data.basic_data['cash_st'].shift(12)/data.basic_data['liab_oth'].shift(12),1,0),index=data.basic_data['liabs'].index, columns=data.basic_data['liabs'].columns) 
        chg_liquid[self.banks] = np.where((data.basic_data['cash_due'][self.banks]+data.basic_data['loan_net'][self.banks]+data.basic_data['receive_int'][self.banks])/(data.basic_data['deposits'][self.banks]+data.basic_data['risk_provision'][self.banks]+data.basic_data['debt'][self.banks])>((data.basic_data['cash_due'][self.banks]+data.basic_data['loan_net'][self.banks]+data.basic_data['receive_int'][self.banks])/(data.basic_data['deposits'][self.banks]+data.basic_data['risk_provision'][self.banks]+data.basic_data['debt'][self.banks])).shift(12),1,0)
        chg_liquid[self.insurers] = np.where((data.basic_data['prem_received'][self.insurers]+data.basic_data['cash_only'][self.insurers])/(data.basic_data['insurance_liab'][self.insurers]+data.basic_data['debt'][self.insurers]+data.basic_data['risk_provision'][self.insurers])>((data.basic_data['prem_received'][self.insurers]+data.basic_data['cash_only'][self.insurers])/(data.basic_data['insurance_liab'][self.insurers]+data.basic_data['debt'][self.insurers]+data.basic_data['risk_provision'][self.insurers])).shift(12),1,0)
        # note - there are some other variations here!!
        chg_neqiss = pd.DataFrame(np.where(data.basic_data['com_shs_out']-data.basic_data['com_shs_out'].shift(12)>1,1,0),index=data.basic_data['com_shs_out'].index, columns=data.basic_data['com_shs_out'].columns)
        chg_roa = pd.DataFrame(np.where(data.basic_data['roa']>data.basic_data['roa'].shift(12),1,0),index=data.basic_data['roa'].index, columns=data.basic_data['roa'].columns)
        chg_fcfta = pd.DataFrame(np.where(data.basic_data['free_cf']>data.basic_data['free_cf'].shift(12),1,0),index=data.basic_data['free_cf'].index, columns=data.basic_data['free_cf'].columns)
        
        chg_margin = pd.DataFrame(np.where(data.basic_data['gross_mgn']>data.basic_data['gross_mgn'].shift(12),1,0),index=data.basic_data['gross_mgn'].index, columns=data.basic_data['gross_mgn'].columns)
        chg_margin[self.banks] = np.where(data.basic_data['income_after_prov'][self.banks]/data.basic_data['int_inc'][self.banks]>(data.basic_data['income_after_prov'][self.banks]/data.basic_data['int_inc'][self.banks]).shift(12),1,0)
        chg_margin[self.insurers] = np.where(data.basic_data['op_inc_preint'][self.insurers]/data.basic_data['sales'][self.insurers]>(data.basic_data['op_inc_preint'][self.insurers]/data.basic_data['sales'][self.insurers]).shift(12),1,0)
        #chg_margin[self.insurers] = np.where(data.basic_data['op_inc_preint'][self.insurers]/data.basic_data['sales'][self.insurers]>(data.basic_data['op_inc_preint'][self.insurers]/data.basic_data['sales'][self.insurers]).shift(12),1,0)
        
        # some variations here too - need to check - for insurers
        chg_turn = pd.DataFrame(np.where(data.basic_data['asset_turn']>data.basic_data['asset_turn'].shift(12),1,0),index=data.basic_data['asset_turn'].index, columns=data.basic_data['asset_turn'].columns)
        
        p_fs = (roa+fcfta+accrual+chg_lever+chg_roa+chg_liquid+chg_neqiss+chg_fcfta+chg_margin+chg_turn)/8.0 #
        return p_fs
        
class QV_fin(QV):
    
    def __init__(self, banks, insurers, threshold=1000, scr2_perc=0.95, scr3_perc = 0.15, scr4_perc = 0.6, upper_limit=0.2):
        #self.rebalance = rebalance
        self.threshold = threshold
        self.scr2_perc = scr2_perc # forensics -> positions 2
        self.scr3_perc = scr3_perc # value -> positions 3
        self.scr4_perc = scr4_perc # quality -> positions 4
        self.upper_limit = upper_limit
        self.banks = banks
        self.insurers = insurers      
    
    def backtest(self, init_pos, data):
        
        self.positions1 = bool_screen(init_pos, self.market_cap(data), threshold=self.threshold) # mkt cap of R2bn or more

        #self.positions2a = screen(self.positions1, self.accruals(data), scr_perc = self.scr2_perc, ascending=True)
    
        #self.positions2b = screen(self.positions1, self.pman(data), scr_perc = self.scr2_perc, ascending=True)
    
        #self.positions2c = screen(self.positions1, self.pfd(data), scr_perc = self.scr2_perc, ascending=True)
    
        #self.positions2 = self.positions2a*self.positions2b*self.positions2c # combine the results of pfd & accrualspositions2 = screen(positions1, data_for_scr2, scr_perc = scr2_perc, ascending=scr2_asc)
    
        self.positions3 = screen(self.positions1, self.value(data), scr_perc = self.scr3_perc, ascending=True)
    
        self.positions4 = screen(self.positions3, self.quality(data), scr_perc = self.scr4_perc, ascending=False)

        self.positions5 = weight(self.positions4, self.mvi_weight(data))

        self.final_positions = limit_pos_size(self.positions5, self.upper_limit)
        
        
class QM(Strategy):
    
    def __init__(self, threshold=1000, scr2_perc = 0.15, scr3_perc = 0.6, upper_limit=0.2):
        self.threshold = threshold
        self.scr2_perc = scr2_perc # momentum
        self.scr3_perc = scr3_perc # fip - quality of momentum  
        self.upper_limit = upper_limit
        
    def backtest(self, init_pos, data):     
        mc= Mkt_cap_scr(threshold=self.threshold)
        self.positions1 = mc.run(init_pos, data) #bool_screen(init_pos, self.market_cap(data), self.threshold) # mkt cap of R2bn or more threshold = 2000
        
        m=Mom(scr_perc = self.scr2_perc)
        self.positions2 = m.run(self.positions1, data)
        
        f=Fip(scr_perc = self.scr3_perc)
        self.positions3 = f.run(self.positions2, data)

        self.positions4 = weight(self.positions3, self.mvi_weight(data))

        self.final_positions = limit_pos_size(self.positions4)
        
        return self.final_positions

class Mom(Strategy):
    
    def __init__(self, scr_perc = 0.15):
        self.scr_perc=scr_perc 
        
    def run(self, init_pos, data):     
        mom = data.basic_data['price_monthly'].pct_change(11).shift(1) #pd.rolling_apply(basic_data['price_monthly'].pct_change(), 12, lambda x: np.prod(1 + x) - 1)
        #mom = data_clean(mom)      
        self.final_positions = screen(init_pos, mom, scr_perc = self.scr_perc, ascending=False)
        return self.final_positions
     
class Fip(Strategy):
    
    def __init__(self, scr_perc = 0.6):
        self.scr_perc = scr_perc # fip - quality of momentum  
        
    def run(self, init_pos, data):     
        fip = np.sign(data.basic_data['price_monthly'].pct_change(11).shift(1))*pd.rolling_apply( data.daily_price.pct_change(), 252, lambda x: (len(np.where(x<0)[0])-len(np.where(x>0)[0]))/252.0)
        fip = fip.reindex(index=data.index, method='ffill')   
        self.final_positions = screen(init_pos, fip, scr_perc = self.scr_perc, ascending=True)      
        return self.final_positions 
        
class Kl_str(Strategy):
    
    def __init__(self, scr_perc = 0.15):
        self.scr_perc=scr_perc
        
    def run(self, init_pos, data):     
        kl_str = 0.7*data.daily_price.pct_change(13*5)+0.3*data.daily_price.pct_change(26*5)
        kl_str = (kl_str.rank(axis=1, ascending = True).T/kl_str.count(axis=1)).T # take percentile
        kl_str= kl_str.reindex(index=data.index, method='nearest')      
        self.final_positions = screen(init_pos, kl_str, scr_perc = self.scr_perc, ascending=False)
        return self.final_positions
        
class Kl_con(Strategy):
  
    def __init__(self, scr_perc = 0.6):
        self.scr_perc=scr_perc
        
    def run(self, init_pos, data):     
        kl_str_obj = Kl_str()        
        kl_con = 2*kl_str_obj.run(init_pos, data) + 2*kl_str_obj.run(init_pos, data).shift(1*5)+1.75*kl_str_obj.run(init_pos, data).shift(2*5)+1.75*kl_str_obj.run(init_pos, data).shift(3*5)+1.5*kl_str_obj.run(init_pos, data).shift(4*5)+1.5*kl_str_obj.run(init_pos, data).shift(5*5)+1.25*kl_str_obj.run(init_pos, data).shift(6*5)+1.25*kl_str_obj.run(init_pos, data).shift(7*5)+1*kl_str_obj.run(init_pos, data).shift(8*5)+1*kl_str_obj.run(init_pos, data).shift(9*5)
        # = pd.rolling_mean(data.kl_str(), 10) 
        kl_con = (kl_con.rank(axis=1, ascending = True).T/kl_con.count(axis=1)).T # take percentile
        kl_con= kl_con.reindex(index=data.index, method='nearest')     
        self.final_positions = screen(init_pos, kl_con, scr_perc = self.scr_perc, ascending=False)
        return self.final_positions
        
class Kl_qual(Strategy):
        
    def __init__(self, scr_perc = 0.6):
        self.scr_perc=scr_perc
        
    def run(self, init_pos, data):     
        weekly_price = data.daily_price.asfreq('W','nearest')        
        kl_qual = pd.rolling_mean(weekly_price.pct_change(1), 40)/pd.rolling_std(weekly_price.pct_change(1), 40) 
        kl_qual = (kl_qual.rank(axis=1, ascending = True).T/kl_qual.count(axis=1)).T# take percentile
        kl_qual= kl_qual.reindex(index=data.index, method='nearest')    
        self.final_positions = screen(init_pos, kl_qual, scr_perc = self.scr_perc, ascending=False)
        return self.final_positions

class Mkt_cap_scr(Strategy):
    
    def __init__(self, threshold=1000):
        self.threshold=threshold 
        
    def run(self, init_pos, data):     
        self.final_positions = bool_screen(init_pos, data.basic_data['market_val'], self.threshold) # mkt cap of R2bn or more threshold = 2000
        return self.final_positions       

class Mkt_cap_weights(Strategy):
        
    def run(self, init_pos, data):     
        self.final_positions = weight(init_pos, self.mkt_weight(data))
        return self.final_positions 

class Mvi_weights(Strategy):
        
    def run(self, init_pos, data, mvi_window_len=220):     
        stdev = pd.rolling_std(data.daily_price, window=mvi_window_len)
        mvi_weight = 1/stdev #daily_price/stdev - this is the volatility indicator
        #test=daily_price.diff()
        #downside_dev = pd.rolling_apply(test, 110, lambda x: np.sqrt((x[x<0]-x.mean())**2).sum()/len(x[x<0]) )
        mvi_weight= mvi_weight.reindex(index=data.index, method='nearest')        
        self.final_positions = weight(init_pos, mvi_weight)
        return self.final_positions
        
class BAH(Strategy):
    def __init__(self):
        pass
        
    def backtest(self, init_pos, data):     
        
        self.final_positions = weight(init_pos, self.mkt_weight(data))
        
        return self.final_positions

class Random_screen(Strategy):
        
    def __init__(self, scr_perc = 0.15):
        self.scr_perc=scr_perc
        
    def run(self, init_pos):     
        self.final_positions = random_screen(init_pos, scr_perc = self.scr_perc)
        return self.final_positions
        
def plot_print(ret, label):
    cum_ret = np.cumsum(ret)
    plt.plot(cum_ret, label=label)


class Channeling(Strategy):
        
    def __init__(self, scr_perc = 0.15):
        self.scr_perc=scr_perc
        
    def run(self, init_pos):     
        will_r = talib.WILLR(high, low, close)
        self.final_positions = np.where(will_r>close.shift(1),1,0) # something like this - this is very basic
        return self.final_positions
