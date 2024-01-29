import pickle
import inflection
import pandas as pd
import numpy as np
import math
import datetime
import json


class Rossmann( object ):
    def __init__( self ):
        self.home_path=''
        self.competition_distance_scaler       = pickle.load( open( self.home_path + 'parameter/competition_distance_scaler.pkl', 'rb' ) )
        self.year_scaler                       = pickle.load( open( self.home_path + 'parameter/year_scaler.pkl', 'rb' ) )
        self.competition_time_in_months_scaler = pickle.load( open( self.home_path + 'parameter/competition_time_in_months_scaler.pkl', 'rb' ) )
        self.promo_time_week_scaler            = pickle.load( open( self.home_path + 'parameter/promo_time_week_scaler.pkl', 'rb' ) )
        self.store_type_encoding               = pickle.load( open( self.home_path + 'parameter/store_type_encoding.pkl', 'rb' ) )
        
        
        
    
    def data_cleaning( self, df1 ):
        # 1 DATA CLEANING
    
        ## 1.1 Rename Columns

        cols_old = df1.columns

        snakecase = lambda x: inflection.underscore( x )

        cols_new = list( map( snakecase, cols_old ) )
        
        # rename
        df1.columns = cols_new

        

        ## 1.3 Data Types

        # Reassigning the column 'date' as datetime.
        df1['date'] = pd.to_datetime( df1['date'] )



        ## 1.5 Fillout NA

        # competition_distance: distance in meters to the nearest competitor store

        # Hypothesis:
        # 1. There is no competitor;
        # 2. They are too far away.

        # Solution #01:
        # Change missing data to a distance waaay bigger than any other competitor distance.

        # Take max distance:
        max_dist = df1['competition_distance'].max()

        # Make the NA values significantly bigger distance than max_dist (times 3 in this case).
        df1['competition_distance'] =  df1['competition_distance'].apply( lambda x: (3*max_dist) if math.isnan( x ) else x )


        # competition_open_since_month/year: gives the approximate month/year of the time the nearest competitor was opened

        # Hypothesis:
        # 1. Competition opened store before us;
        # 2. It was simply not record.

        # Solution #01:
        # Use the column 'date' as reference for the opening of competition and watch the impact on the ML algorithms.
        df1['competition_open_since_month'] = df1.apply( lambda x: x['date'].month if math.isnan( x['competition_open_since_month'] ) else x['competition_open_since_month'], axis=1 )
        df1['competition_open_since_year'] = df1.apply( lambda x: x['date'].year if math.isnan( x['competition_open_since_year'] ) else x['competition_open_since_year'], axis=1 )


        # promo2_since_week/year: describes the year and calendar week when the store started participating in Promo2

        # Hypothesis:
        # 1. promo2 == 0.

        # Check hypothesis:
        # df1.loc[ df1['promo2'] == 0, ['promo2_since_week', 'promo2_since_year'] ].isna().sum()
        # Looks perfect!

        # Solution #01:
        # Fill NA with 'date' values.
        df1['promo2_since_week'] = df1.apply( lambda x: x['date'].week if math.isnan( x['promo2_since_week'] ) else x['promo2_since_week'], axis=1 )
        df1['promo2_since_year'] = df1.apply( lambda x: x['date'].year if math.isnan( x['promo2_since_year'] ) else x['promo2_since_year'], axis=1 )


        # promo_interval: describes the consecutive intervals Promo2 is started, naming the months the promotion is started anew. 
        # E.g. "Feb,May,Aug,Nov" means each round starts in February, May, August, November of any given year for that store

        # Hypothesis:
        # 1. promo2 == 0

        # Check hypothesis?
        # df1.loc[ df1['promo2'] == 0, [ 'promo_interval'] ].isna().sum()
        # Looks perfect!

        # Solution #01:
        # Fill NA with zeros: no promotion interval.
        df1['promo_interval'] = df1['promo_interval'].fillna( 0 )


        # CHECKING IF CONTINUING PROMOTION ('promo2') IS HAPPENING IN EACH SALE

        # promo_interval: describes the consecutive intervals Promo2 is started, naming the months the promotion is started anew. 
        # E.g. "Feb,May,Aug,Nov" means each round starts in February, May, August, November of any given year for that store

        # Map month of each sale.
        month_map = { 1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec' }
        df1['month_map'] = df1['date'].dt.month.map( month_map )

        # Define if there is a promo in each sale. Create a new column indicating if a sale is done with or without promotion.
        df1['is_promo'] = df1[[ 'promo_interval', 'month_map' ]].apply( lambda x: ( 0 if x['promo_interval'] == 0 
                                                                                    else ( 1 if x['month_map'] in x['promo_interval'].split( ',' ) 
                                                                                           else 0 ) ), axis=1 )
        
        
        
        ## 1.6 Change types

        # Reassign week, month and year columns as integers.
        df1['competition_open_since_month'] = df1['competition_open_since_month'].astype( int )
        df1['competition_open_since_year']  = df1['competition_open_since_year'].astype( int )
        df1['promo2_since_week']            = df1['promo2_since_week'].astype( int )
        df1['promo2_since_year']            = df1['promo2_since_year'].astype( int )

        
        return df1
    
    
    
    
    def feature_engineering( self, df2 ):
        # FEATURE ENGINEERING
        
        # year
        df2['year'] = df2['date'].dt.year

        # month
        df2['month'] = df2['date'].dt.month

        # day
        df2['day'] = df2['date'].dt.day

        # week_of_year
        df2['week_of_year'] = df2['date'].dt.isocalendar().week.astype( int )

        # year_week
        df2['year_week'] = df2['date'].dt.strftime( '%Y-%W' )

        # competition_since
        df2['competition_since'] = df2.apply( lambda x: datetime.datetime( year=x['competition_open_since_year'], month=x['competition_open_since_month'], day=1 ), axis=1 )
        df2['competition_time_in_months'] = ( ( df2['date'] - df2['competition_since'] ) / 30 ).apply( lambda x: x.days ).astype( 'int64' )

        # promo_since
        df2['promo_since'] = df2['promo2_since_year'].astype( str ) + '-' + df2['promo2_since_week'].astype( str )
        # "%Y-%W-%w has week 1 as the week after the year's first sunday.
        df2['promo_since'] = df2['promo_since'].apply( lambda x: datetime.datetime.strptime( x + '-1', "%Y-%W-%w" ) - datetime.timedelta( days=7 ) )
        df2['promo_time_week'] = ( ( df2['date'] - df2['promo_since'] ) / 7 ).apply( lambda x: x.days ).astype( 'int64' )

        # assortment
        df2['assortment'] = df2['assortment'].apply( lambda x: 'basic' if x == 'a' else 'extra' if x == 'b' else 'extended' )

        # state_holiday
        df2['state_holiday'] = df2['state_holiday'].apply( lambda x: 'public_holiday' if x == 'a' else 'easter_holiday' if x == 'b' else 'christmas' if x == 'c' else 'regular_day' )
        
        
        
        # 3 FILTRAGEM DE VARIÁVEIS

        ## 3.1 Filtragem das linhas
        
        df2 = df2[ df2['open'] != 0 ]
        
        
        ## 3.2 Seleção das colunas
        
        # Deletando 'open' porque só tem valor == 1,
        # Deletando 'promo_interval' e 'month_map' porque já foram usada para criar 'is_promo',
        # Deletando 'customers' porque não teremos informação de quantos clientes teremos.
        # Assim como 'customers', checar outras colunas que não existem em test.csv
        cols_drop = [ 'open', 'promo_interval', 'month_map' ]
        df2 = df2.drop( cols_drop, axis=1 )
        
        
        return df2


        

    def data_preparation( self, df5 ):
        # 5 DATA PREPARATION

        ## 5.2 Rescaling
        
        # year
        df5['year'] = self.year_scaler.fit_transform( df5[['year']].values )

        # competition_distance - Robust
        df5['competition_distance'] = self.competition_distance_scaler.fit_transform( df5[['competition_distance']].values )

        # competition_time_in_months
        df5['competition_time_in_months'] = self.competition_time_in_months_scaler.fit_transform( df5[['competition_time_in_months']].values )

        # promo_time_week
        df5['promo_time_week'] = self.promo_time_week_scaler.fit_transform( df5[['promo_time_week']].values )
        
        
        ### 5.3.1 Categorical Encoding
        
        # state_holiday - One Hot Encoding
        df5 = pd.get_dummies(df5, prefix=['state_holiday'], columns=['state_holiday'], dtype=int)

        # store_type - Label Encoding
        df5['store_type'] = self.store_type_encoding.fit_transform( df5['store_type'] )

        # assortment - Ordinal Encoding
        assort_dict = { 'basic': 1, 'extra': 2, 'extended': 3 }
        df5['assortment'] = df5['assortment'].map( assort_dict ) 


        ### 5.3.3 Cyclical Features Encoding

        # CYCLICAL FEATURES ENCODING

        # day_of_week
        df5['dayweek_sin'] = df5['month'].apply( lambda x: np.sin( x * ( 2. * np.pi/7 ) ) )
        df5['dayweek_cos'] = df5['month'].apply( lambda x: np.cos( x * ( 2. * np.pi/7 ) ) )

        # month
        df5['month_sin'] = df5['month'].apply( lambda x: np.sin( x * ( 2. * np.pi/12 ) ) )
        df5['month_cos'] = df5['month'].apply( lambda x: np.cos( x * ( 2. * np.pi/12 ) ) )

        # day
        df5['daymonth_sin'] = df5['month'].apply( lambda x: np.sin( x * ( 2. * np.pi/30 ) ) )
        df5['daymonth_cos'] = df5['month'].apply( lambda x: np.cos( x * ( 2. * np.pi/30 ) ) )

        # week_of_year
        df5['weekyear_sin'] = df5['month'].apply( lambda x: np.sin( x * ( 2. * np.pi/52 ) ) )
        df5['weekyear_cos'] = df5['month'].apply( lambda x: np.cos( x * ( 2. * np.pi/52 ) ) )
        
        
        # 6 FEATURE SELECTION
        
        ## 6.3 Manual feature selection
        
        # boruta features
        cols_selected = [ 'store', 'promo', 'store_type', 'assortment', 'competition_distance', 'competition_open_since_month', 'competition_open_since_year',
                          'promo2', 'promo2_since_week', 'promo2_since_year', 'competition_time_in_months', 'promo_time_week', 'dayweek_sin', 'dayweek_cos', 
                          'month_sin', 'month_cos', 'daymonth_sin', 'daymonth_cos', 'weekyear_sin', 'weekyear_cos' ]
        
        
        return df5 [ cols_selected ]
    
    
    
    
    def get_prediction( self, model, original_data, test_data ):
        # prediction
        pred = model.predict( test_data )
        
        # join pred into the original data
        original_data['prediction'] = np.expm1( pred )
        
        return original_data.to_json( orient='records', date_format='iso' )