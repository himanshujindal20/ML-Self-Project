import pandas as pd
from math import sqrt 
import numpy as np 
import os 


class Strategy: 
    def __init__(self,data):
        
        self.data = data
        self.annual_days = 365

        '''
        Your strategy has to coded in this section 
        A sample strategy of moving average crossover has been provided to you. You can uncomment the same and run the code for checking data output 
        You strategy module should return a signal dataframe
        '''
        
    def strategy(self):

        df=self.data
        df.rename(columns = {'Series1':1,'Series2':2,'Series3':3,'Series4':4,'Series5':5,'Series6':6,'Series7':7,'Series8':8,'Series9':9, 'Series10':10}, inplace = True)

        for i in range(1,11):
            ShortEMA= df[i].ewm( span=12, adjust=False).mean()
            LongEMA=df[i].ewm(span=26, adjust= False).mean()
            MACD = ShortEMA - LongEMA
            signal = MACD.ewm(span=9, adjust=False).mean()
            df['MACD' + str(i)]= MACD
            df['Signal Line' + str(i)]= signal
            df['MA200' +str(i)]= df[i].rolling(window=200).mean()


        for i in range(1,11):
            pricechange= df[i]. pct_change()
            upmove= pricechange.apply(lambda x: x if x>0 else 0)
            downmove=pricechange.apply(lambda x: abs(x) if x<0 else 0)
            avgup= upmove.ewm(span=27).mean()
            avgdown=downmove.ewm(span=27).mean()
            RS= avgup/ avgdown
            df['RSI' + str(i)]= RS.apply(lambda x: (100-(100/(x+1))))


        for j in range(1,11):
            for i in range(0, len(df)):
                if (df['MACD'+str(j)][i] > df['Signal Line' + str(j)][i]) or (df['RSI'+ str(j)][i]>  55 ) :
                    df[j][i]=10
                elif (df[j][i] > df['MA200' +str(j)][i]) and (df['RSI'+ str(j)][i]<  30 ):
                    df[j][i]=10
                else:
                    df[j][i]= -10


        for i in range(1,11):
            columns_to_remove = ['MACD'+str(i), 'RSI'+str(i), 'Signal Line'+str(i), 'MA200'+str(i)]
            df.drop(labels=columns_to_remove,axis=1, inplace=True)


        df.rename(columns = {1:'Series1',2:'Series2',3:'Series3',4:'Series4',5:'Series5',6:'Series6',7:'Series7',8:'Series8',9:'Series9', 10:'Series10'}, inplace = True)

        return df
        
        '''
        This module computes the daily asset returns based on long/short position and stores them in a dataframe 
        '''
    def process(self):
        returns = self.data.pct_change()
        self.signal = self.strategy()
        self.position = self.signal.apply(np.sign)
        self.asset_returns = (self.position.shift(1)*returns)
        return self.asset_returns

        '''
        This module computes the overall portfolio returns, asset portfolio value and overall portfolio values 
        '''

    def portfolio(self):
        asset_returns = self.process()
        self.portfolio_return = asset_returns.sum(axis=1)
        self.portfolio = 100*(1+self.asset_returns.cumsum())
        self.portfolio['Portfolio'] = 100*(1+self.portfolio_return.cumsum())
        return self.portfolio

        '''
        This module computes the sharpe ratio for the strategy
        '''

    def stats(self):
        stats = pd.Series()
        self.index = self.portfolio()
        stats['Start'] = self.index.index[0]
        stats['End'] = self.index.index[-1]
        stats['Duration'] = pd.to_datetime(stats['End']) - pd.to_datetime(stats['Start'])
        annualized_return = self.portfolio_return.mean()*self.annual_days
        stats['Annualized Return'] = annualized_return
        stats['Annualized Volatility'] = self.portfolio_return.std()*sqrt(self.annual_days)
        stats['Sharpe Ratio'] = stats['Annualized Return'] / stats['Annualized Volatility']
        return stats
        
if __name__ == '__main__':

    """ 
    Function to read data from csv file 
    """
    data = pd.read_csv(os.path.join(os.getcwd(),'Data.csv'),index_col='Date')
    result = Strategy(data)
    res = result.stats()
    res.to_csv(os.path.join(os.getcwd(),'Result.csv'),header=False)





