from copyreg import pickle
import pandas as pd
import numpy as np
import plotly.express as xp
import joblib

import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from pandas.plotting import lag_plot
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf

import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.metrics import  mean_squared_error as mse
from  sklearn.metrics import mean_absolute_percentage_error as mape
import warnings
warnings.filterwarnings("ignore")


#read data
kansas_df=pd.read_csv("kansas 2011-2017 temp data.csv")
drug_df_train=pd.read_csv("drugsComTrain_raw.tsv",sep="\t")
drug_df_test=pd.read_csv("drugsComTest_raw.tsv",sep="\t")

#pick kansas state data
kansas_only=kansas_df[kansas_df["name"]=="kansas"]

#sort values
kansas_only["datetime"]=pd.to_datetime(kansas_only["datetime"])
kansas_only.sort_values(by="datetime",inplace=True)
temp_df=kansas_only[["datetime","temp","humidity"]]
temp_df.drop_duplicates(subset=["datetime"],keep="first",inplace=True)
temp_df.reset_index(inplace=True,drop=True)

#drugs data
drug_df_train=drug_df_train[["drugName","condition","date","usefulCount"]]
drug_df_train["date"]=pd.to_datetime(drug_df_train["date"])
drug_df_test["date"]=pd.to_datetime(drug_df_test["date"])
drug_df_train.sort_values(by="date",inplace=True)

#create main dataframe
main_df=pd.merge(drug_df_train,temp_df,left_on=["date"],right_on=["datetime"],how="left")

#create new columns
main_df["year"]=main_df["date"].dt.year
main_df["month"]=main_df["date"].dt.month
main_df["week"]=main_df["date"].dt.week

#sort and drop missing values
main_df.sort_values(by="date",inplace=True)
main_df.dropna(how="any",inplace=True)

#cumulative features
#get cumulative stats per week i.e total count of drugs per week and average temperature and humidity per month
drug_sum=main_df.groupby(["drugName","year","week"]).sum().reset_index().rename(columns={"usefulCount":"total_weekly_stock"})[["drugName","year","week","total_weekly_stock"]]
humidity_temp=main_df.groupby(["year","week"]).mean().reset_index().rename(
    columns={"temp":"avg_wk_temp","humidity":"avg_wk_humidity"})[["year","week","avg_wk_temp","avg_wk_humidity"]]
main_df_agg2=pd.merge(drug_sum,humidity_temp,how="left",on=["year","week"])

#reset_index for each drug name
main_df_agg2=main_df_agg2.groupby(["drugName"]).apply(lambda x : x.reset_index(drop=True).drop(columns="drugName")).reset_index()

#dates for weeks
#lets get the start date for week..since we will be forecasting weekly values
dates_df=pd.DataFrame({"dates":pd.date_range(start='2011-01-01', end='2018-01-01',freq='1D')})
dates_df["year"]=dates_df["dates"].dt.year
dates_df["week"]=dates_df["dates"].dt.week
min_week_date=dates_df.groupby(["year","week"]).agg({"dates":"min"}).reset_index().rename(columns={"dates":"min_week_date"})

#merge with date
main_df_agg2=pd.merge(main_df_agg2,min_week_date,on=["year","week"],how="left")


clean_df=main_df_agg2[(main_df_agg2["year"]>=2014) & (main_df_agg2["min_week_date"]<pd.Timestamp(2017,10,1))]
clean_df.sort_values("min_week_date",inplace=True)

#models
#fit SARIMAX model


def SARIMAX_model(stock_vals,forecast_len,param_1):
    """
    Forecast using SARIMAX model
    """
    SARIMAX_model= SARIMAX(stock_vals, order=(1, 1,3), seasonal_order=(1, 1, 1,param_1)
                             ).fit(dis=-1)
    forecast_vals_SARIMAX=SARIMAX_model.get_forecast(steps=forecast_len)
    predicted=forecast_vals_SARIMAX.summary_frame()["mean"].values
    return predicted

def ARIMA_model(stock_vals,forecast_len):
    """
    Forecast using ARIMA model
    """
    ARIMA_model= ARIMA(stock_vals,order=(3,1,3))
    ARIMA_model_fit = ARIMA_model.fit()
    result = ARIMA_model_fit.forecast(forecast_len, alpha=0.10)
    return result


#norgest drgu dataframe
drug_variable="Ethinyl estradiol / norgestimate"
drug_df_1=clean_df[clean_df["drugName"]==drug_variable][["drugName","year","week",
                                                            "total_weekly_stock","min_week_date"]]

forecast_len=6#next six weeks
drug_df_1_train=drug_df_1.head(drug_df_1.shape[0]-forecast_len)
drug_df_1_test=drug_df_1.tail(forecast_len)

#call model functions
train_vals=drug_df_1_train["total_weekly_stock"]
next_wks_SARIMAX=SARIMAX_model(train_vals,forecast_len,6)
next_wks_ARIMA=ARIMA_model(train_vals,forecast_len)

#get results
drug_df_1_test["predicted_SARIMAX"]=next_wks_SARIMAX
drug_df_1_test["predicted_ARIMA"]=next_wks_ARIMA


train_test_1=pd.concat([drug_df_1_train,drug_df_1_test])
last_six_week=train_test_1.tail(6)
print(last_six_week)

fig=xp.line(train_test_1,x="min_week_date",y=["total_weekly_stock","predicted_SARIMAX","predicted_ARIMA"],
           title=" Full time series Ethinyl estradiol / norgestimate with actual and predicted")
fig.show()

figure=xp.pie(train_test_1,"min_week_date","total_weekly_stock","predicted_SARIMAX","predicted_ARIMA",
           title=" Full time series Ethinyl estradiol / norgestimate with actual and predicted")

figure.show()

joblib.dump(SARIMAX_model, 'SARIMAX.pkl')
print(SARIMAX_model.predict([[4, 300, 500]]))
