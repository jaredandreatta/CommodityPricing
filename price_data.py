import yfinance as yf
import pandas as pd

'''
This model is specifically made for the analysis of energy and metal commodity
futures using the yfinance API. I will include a list of valid inputs, though
all tickers will count as a valid input.
'''
valid_inputs_dict= {
'Commodity': ['Gold', 'Silver', 'Platinum', 'Copper', 'Palladium', 'Crude Oil', 
            'Heating Oil', 'Natural Gas', 'RBOB Gasoline'],
'Input Value': ['GC=F', 'SI=F', 'PL=F', 'HG=F', 'PA=F', 'CL=F', 'HO=F', 'NG=F',
            'RB=F']
}

def get_price_data(commodity, start_date='2000-1-1', end_date='2024-06-01'):
    '''
    Fetches price data on a selected commodity via the yfinance API.

    Parameters:
    commodity (str): desired commodity for analysis.
    start_date(str): must be in yyyy-m-d format
    end_date(str): must be in yyyy-m-d format

    Returns:
    pd.DataFrame: Dataframe including historical commodity prices
    *** Tables of valid arguments will be included ***

    '''
    
    try:
        df = yf.download(commodity.upper(), start_date, end_date)['Adj Close']
        df=df.reset_index()
        df.columns=['Date','Price']
        return df
    except ValueError:
        print('Invalid ticker name. Call gpd_help() for help on valid inputs.')
    except Exception as e:
        print(f'An error occurred: {e}')
        print('Invalid inputs. Call gpd_help() for help on valid inputs.')
    
def gpd_help():
    print("List of valid ticker inputs:")
    return print(pd.DataFrame.from_dict(valid_inputs_dict))
    
   