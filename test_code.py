import pandas as pd
import numpy as np
from IPython.display import display
from datetime import datetime
from dateutil.relativedelta import relativedelta
import os
import random
import warnings
import itertools
import statsmodels.api as sm
import matplotlib.pyplot as plt


def frame_converter(sample_data):
    dates = sample_data.columns.values.tolist()
    product_id_column = dates.pop(0)
    values = sample_data.values.tolist()[0]
    product_id = values.pop(0)
    converted_frame = pd.DataFrame({'Date': dates, 'Sales': values})
    converted_frame['Date'] = pd.to_datetime(converted_frame['Date'])
    converted_frame = converted_frame.set_index('Date')
    return converted_frame


def arima_forecasting(book, number_of_months, pre_predict_months):
    # Time series forecasting with ARIMA
    warnings.filterwarnings("ignore")
    book = book.iloc[:-pre_predict_months]
    y = book['Sales'].resample('MS').mean()

    p = d = q = range(0, 2)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
    param_dict = {}

    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(y, order=param, seasonal_order=param_seasonal,
                                                enforce_stationarity=False, enforce_invertibility=False)

                results = mod.fit()

                param_dict[param, param_seasonal] = results.aic
            except:
                continue

    param = min(param_dict, key=param_dict.get)[0]
    param_seasonal = min(param_dict, key=param_dict.get)[1]

    # Fitting the ARIMA model & Validating forecasts
    data_end = book.index[-1]
    start_date = (data_end + relativedelta(months=1)).strftime('%Y-%m-%d')
    end_date = (data_end + relativedelta(months=number_of_months + pre_predict_months)).strftime('%Y-%m-%d')

    mod = sm.tsa.statespace.SARIMAX(y, order=param, seasonal_order=param_seasonal, enforce_stationarity=False,
                                    enforce_invertibility=False)
    results = mod.fit()
    pred = results.get_prediction(start=pd.to_datetime(start_date), end=pd.to_datetime(end_date), dynamic=True)
    pred_ci = pred.conf_int()
    y_forecasted = pred.predicted_mean

    arima_forecast = pd.DataFrame({'date': y.index, 'arima_prediction': y.values}).append(
        pd.DataFrame({'date': y_forecasted.index, 'arima_prediction': y_forecasted.values})).reset_index(drop=True)
    return arima_forecast


def time_series_modelling(dataframe):
    for row in range(0, len(dataframe)):
        product_name = dataframe.iloc[row][0]
        directory = 'time_series_results/' + product_name
        if not os.path.exists(directory): os.makedirs(directory)
        sample_data = pd.DataFrame(dataframe.iloc[[row]], columns=dataframe.columns.values)
        book = frame_converter(sample_data)
        book.index.names = ['Date']
        book.columns = ['Sales']
        number_of_months = 24
        pre_predict_months = 12
        arima_forecast = arima_forecasting(book, number_of_months, pre_predict_months)
        combo_df = arima_forecast.copy()
        sales = book['Sales'].values.tolist()
        sales.extend([np.nan]*(len(combo_df['date']) - len(book['Sales'])))
        final_df = pd.DataFrame()
        final_df['date'] = combo_df['date']
        final_df['sales'] = sales
        combo_df = combo_df[combo_df['date'] >= (book.index[-pre_predict_months] + relativedelta(months=1)).strftime(
            '%Y-%m-%d')].reset_index(drop=True)
        final_df = pd.merge(final_df, combo_df, how='left', on=['date'], sort=False)
        final_df.plot(x='date', y=['sales', 'arima_prediction'], style=['-', '--'], color=['blue', 'green'],
            figsize=(15, 8))
        final_df.to_csv(directory + '/' + product_name + '.csv', index=False)
        plt.savefig(directory + '/' + product_name + '.png')
        plt.close()


def time_series_modelling_wrapper(csv_file):
    dataframe = pd.read_csv(csv_file)
    time_series_modelling(dataframe)


time_series_modelling_wrapper('masked_data_shared.csv')
