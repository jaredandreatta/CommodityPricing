import yfinance as yf
import pandas as pd

'''
This model is specifically made for the analysis of energy and metal commodity
futures using the yfinance API. I will include a list of valid inputs, though
all tickers will count as a valid input.
'''


def get_price_data(commodity, start_date='2000-1-1', end_date='2024-06-01'):
    '''
    PARAMS:

    commodity: str of desired commodity for analysis.
    start_date: str (must be in yyyy-m-d format)
    end_date: str (must be in yyyy-m-d format)

    *** Tables of valid arguments will be included ***

    '''
    
    try:
        df = yf.download(commodity, start_date, end_date)['Adj Close']
        return df
    except:
        print(f'An error occurred: {e}')
        print('Invalid inputs. Call gpd_help() for help on valid inputs.')
    
def gpd_help():
    #valid_inputs_dict= {
    #'Commodity Futures': ['Gold', 'Silver', 'Platinum', 'Copper', 'Palladium', 'Crude Oil', 
    #            'Heating Oil', 'Natural Gas', 'RBOB Gasoline'],
    #'Ticker': ['GC=F', 'SI=F', 'PL=F', 'HG=F', 'PA=F', 'CL=F', 'HO=F', 'NG=F',
    #           'RB=F']
    #}
    #return pd.DataFrame.from_dict(valid_inputs_dict)
    return None