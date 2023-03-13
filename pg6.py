
"""Name: Tenzing Nyima
Email: Tenzing.Nyima71@myhunter.cuny.edu

"""
import math
import pickle
import pandas as pd
#import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score


def import_data(file_name):
    """The data in the file is read into data frame
    and the columns: VendorID, RatecodeID, store_and_fwd_flag, payment_type,
    extra, mta_tax, tolls_amount, improvement_surcharge,
    congestion_surcharge, are dropped
    any rows with non-positive total_amount are dropped"""
    df = pd.read_csv(file_name)
    df = df.drop(columns={'VendorID','RatecodeID', 'store_and_fwd_flag',
                          'payment_type','extra','mta_tax','tolls_amount',
                          'improvement_surcharge','congestion_surcharge'})
    print(df)
    df = df[df['total_amount'] > 0]
    #df['percent_tip'] = df['tip_amount']* 100/df['total_amount']
    #print(df)
    return df

def add_tip_time_features(df):
    """This function takes one input
    computes 3 new columns
    percent tip:
    duration : the time the trip took
    dayofweek: """
    df['percent_tip'] = df['tip_amount']* 100/df['total_amount']
    df['pickup_time'] = pd.to_datetime(df['tpep_pickup_datetime'])
    df['dropoff_time'] = pd.to_datetime(df['tpep_dropoff_datetime'])
    df['duration'] = (df['dropoff_time'] - df['pickup_time']).dt.seconds
    df['dayofweek'] = df['pickup_time'].dt.dayofweek
    df = df.drop(columns=['pickup_time','dropoff_time'])
    return df

def impute_numeric_cols(df):
    """Missing data in the numeric columns passenger_count, trip_distance, fare_amount
    , tip_amount, total_amount, duration, dayofweek are replaced with the median of
    respective coumn"""
    #deremine if any value in a series is missing.
    missing_cols = df.columns[df.isnull().any()]
    print(missing_cols)
    for cols in missing_cols:
        median_= df[cols].median(skipna= True)
        if df[cols].isnull().values.any():
            df[cols] = df[cols].replace(math.nan, median_)
    return df

def add_boro(df, file_name)-> pd.DataFrame:
    """Makes a datafrme , using file_name, to add pick up and drop off borough to df.
    in particular, adds two new columns to the df
    PU_borough: that contin the borough correspoinding to pick up taxi zope ID
    DO_borough that contain the borough corresponding to teh drop off taxi zone.
    returns df with two additional columns.
    """
    print(df)
    df_borough = pd.read_csv(file_name)
    df_borough = df_borough.rename(columns={'LocationID':'PULocationID'})
    df_borough = df_borough.drop(columns={'zone','Shape_Area','the_geom','OBJECTID','Shape_Leng'})
    print(df_borough)
    df = pd.merge(left=df,right=df_borough, how="left", on="PULocationID")
    df = df.rename(columns={'borough': 'PU_borough'})
    df_borough = df_borough.rename(columns={'PULocationID':'DOLocationID'})
    df = pd.merge(left = df, right=df_borough,how="left", on="DOLocationID")
    df = df.rename(columns={'borough':'DO_borough'})
    print(df)
    return df

def encode_categorical_col(col, prefix):
    """Takes a column of categorical data and uses categorical encoding
    to create a dataframe with k-1 columns, where k is the number of
    different nominal values of column. Your function should create k
    columns, one for each value, labels by prefix concatenated with teh
    value. The columns hsould be sored and the Data frme restricted to the
    first k-1 columns retured."""
    df = pd.get_dummies(col)
    print(df)
    df = df.iloc[:, :-1]
    df = df.add_prefix(prefix)
    df = df[sorted(df)]
    return df


def split_test_train(df, xes_col_names,y_col_name, test_size=0.25, random_state=1870):
    """Takes 5 input aparaemters,
    df
    y_col_name:
    xes_col_names: a list of coumns tha tcontain the independent variabels
    test_size: accepts a float between 0 and 1 adn represents teh proportion of the
    data set to use for training. This parameter has a default value of 0.25.
    random_state: Used as a seed to teh randomization. This parameter has a
    default value of 1870."""
    x_col = df.filter(xes_col_names)
    y_col = df[y_col_name]
    x_train, x_test, y_train, y_test = train_test_split(x_col,y_col, test_size, random_state)
    return x_train, x_test, y_train, y_test

def fit_linear_regression(x_train, y_train):
    """Fits a linear model to x_tain and y_train, using sklearn.linear_model.LinearRegression
    (see lecture and textbook for details on setting up the model).
    The resulting model should be returned as bytestream, using pickle"""
    reg = linear_model.LinearRegression().fit(x_train,y_train)
    p_mod = pickle.dump(reg)
    return p_mod

def predict_using_trained_model(mod_pkl, xes, yes):
    """This function takes there inputs
    mod_pkl: an array or data, stored in pickle format , 
    xes: an array of dataframe of numeric columns with no null values, 
    yes: an array or dataframe of numeric columsn with no null values. 
    Computes and retuns the mean squared error and r2 socre between values predicted by the model 
    and actual values (y), Not taht sklesrn. metric contain two function taht may of of use. 
    mean_squared_erro adn r2_score."""    
    mod = pickle.loads(mod_pkl)
    predict_y = mod.predict(xes)
    mse  = mean_squared_error(predict_y, yes)
    print(mse)
    r = r2_score(predict_y, yes)
    return mse, r

def main():
    df = import_data('taxi_jfk_june2020.csv')
    df = add_tip_time_features(df)
    print(df[ ['trip_distance','duration','dayofweek','total_amount','percent_tip'] ].head() )
    print(df[ ['passenger_count','trip_distance'] ].head(10) )
    df = impute_numeric_cols(df)
    print( df[ ['passenger_count','trip_distance'] ].head(10) )
    #Explore some data:
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(color_codes=True)

    sns.lmplot(x="total_amount", y="percent_tip", data=df)
    tot_r = df['total_amount'].corr(df['percent_tip'])
    plt.title(f'Taxi Trips from JFK, June 2020 with r = {tot_r:.2f}')
    plt.tight_layout()  #for nicer margins
    plt.show()
    sns.lmplot(x="trip_distance", y="percent_tip", data=df)
    dist_r = df['trip_distance'].corr(df['percent_tip'])
    plt.title(f'Taxi Trips from JFK, June 2020 with r = {dist_r:.2f}')
    plt.tight_layout()  #for nicer margins
    plt.show()
    df = add_boro(df,'taxi_zones.csv')
    print('\nThe locations and borough columns:')
    print(f"{df[['PULocationID','PU_borough','DOLocationID','DO_borough']]}")
    df_do = encode_categorical_col(df['DO_borough'],'DO_')
    print(df_do.head())
    df_all = pd.concat( [df,df_do], axis=1)
    print(f'The combined DataFrame has columns: {df_all.columns}')
    df_all = impute_numeric_cols(df_all)
    num_cols = ['passenger_count','trip_distance','fare_amount',\
    'tip_amount', 'total_amount', 'percent_tip', 'duration','dayofweek',\
    'PU_borough', 'DO_borough', 'DO_Bronx', 'DO_Brooklyn','DO_EWR',\
    'DO_Manhattan', 'DO_Queens']
    x_train,x_test,y_train,y_test = split_test_train(df_all,num_cols, 'percent_tip')
    print('For numeric columns, training on 25% of data:')
    mod_pkl = fit_linear_regression(x_train,y_train)
    mod = pickle.loads(mod_pkl)
    print(f'intercept = {mod.intercept_} and coefficients = {mod.coef_}')
    tr_err,tr_r2 = predict_using_trained_model(mod_pkl, x_train,y_train)
    print(f'training:  RMSE = {tr_err} and r2 = {tr_r2}.')
    test_err,test_r2 = predict_using_trained_model(mod_pkl, x_test,y_test)
    print(f'testing:  RMSE = {test_err} and r2 = {test_r2}.')
    x_train,x_test,y_train,y_test = split_test_train(df_all,num_cols, 'percent_tip')
    print('For numeric columns, training on 25% of data:')
    mod_pkl = fit_linear_regression(x_train,y_train)
    mod = pickle.loads(mod_pkl)
    print(f'intercept = {mod.intercept_} and coefficients = {mod.coef_}')
    tr_err,tr_r2 = predict_using_trained_model(mod_pkl, x_train,y_train)
    print(f'training:  RMSE = {tr_err} and r2 = {tr_r2}.')
    test_err,test_r2 = predict_using_trained_model(mod_pkl, x_test,y_test)
    print(f'testing:  RMSE = {test_err} and r2 = {test_r2}.')
    print(f'Prediction for 4 July data with only duration and total amount:')
    df_july = import_data('program06/taxi_4July2020.csv')
    df_july = add_tip_time_features(df_july)
    df_july = impute_numeric_cols(df_july)
    print(df_july[['duration','total_amount']])
    july_err,july_r2 = predict_using_trained_model(mod2_pkl, df_july[['duration','total_amount']].to_numpy(),df_july['percent_tip'])
    print(f'RMSE = {july_err} and r2 = {july_r2}.')
    print(f'Prediction for 4 July data with full model:')
    df_july = add_boro(df_july,'program06/taxi_zones.csv')
    df_do_j = encode_categorical_col(df_july['DO_borough'],'DO_')
    df_all_j = pd.concat( [df_july,df_do_j], axis=1)
    july_err,july_r2 = predict_using_trained_model(mod_pkl, df_all_j[num_cols].to_numpy(),df_all_j['percent_tip'])
    print(f'RMSE = {july_err} and r2 = {july_r2}.')
    
main()