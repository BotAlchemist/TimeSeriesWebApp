# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 08:14:40 2021

@author: Sumit
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from statsmodels.tsa.vector_ar.var_model import VAR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import os


st.title('Time Series Forecasting Web App')

#run_algorithm= 0

files= [f for f in os.listdir('.') if os.path.isfile(f)]
files= list(filter(lambda f: f.endswith('.csv'), files))

filename_select = st.sidebar.selectbox('Select dataset', files)

df= pd.read_csv(filename_select)


date_column= st.sidebar.selectbox('Select date column', df.columns.tolist())
target_column= st.sidebar.selectbox('Select target column', df.columns.tolist())

ts_algorithm_name_list= ['Null', 'VAR', 'K Nearest Neighbor', 'Random Forest', 'ARIMA']
ts_algorithm_name= st.sidebar.selectbox('Select algorithm', ts_algorithm_name_list)




def display_results(train_df, test_df, target_column):
    st.markdown("### Plot Actual vs Predicted")
    st.line_chart(test_df[['Prediction', target_column]])
            
    st.markdown("### Dataset Actual vs Predicted")
    st.write(test_df[[target_column, 'Prediction']])
            
    MSE= mean_squared_error(test_df[target_column], test_df['Prediction'])
    RMSE= MSE**0.5
    st.markdown("#### Root Means Squred Error: "+ str(round(RMSE,2)))
            
    st.markdown("### Overall graph")
    train_df= train_df.append(test_df)
    st.line_chart(train_df[['Prediction', target_column]])
    

def run_auto_var(data, n):
    best_aic = np.inf
    best_p = None
    tmp_model = None
    best_mdl = None
    for p in range(n+1):
        try:
            
            model = VAR(data)
            model_fit = model.fit(p)
            if model_fit.aic < best_aic:
                best_aic = model_fit.aic
                best_p = p
        except:
            continue
    return best_p, best_aic


def get_best_k(train, test, target_column):
    best_RMSE= np.inf
    for k in range(1, 10):
        knn= KNeighborsRegressor(n_neighbors= k)
        knn.fit(train.drop(target_column,axis=1),train[target_column])
        y_pred=knn.predict(test.drop([target_column], axis=1))
        y_pred= [round(num) for num in y_pred]
        MSE = mean_squared_error(test[target_column], y_pred)
        RMSE = round(MSE**0.5,2)
        
        if RMSE <=best_RMSE:
            best_RMSE= RMSE
            best_k= k
    return best_k

if len(filename_select) !=0  and date_column != target_column and ts_algorithm_name != 'Null':
    run_algorithm= st.sidebar.checkbox("Run Algorithm")
    if run_algorithm:
        st.markdown("### Dataset")
        df[date_column]= pd.to_datetime(df[date_column])
        df['Week_name'] = df[date_column].dt.week
        df['Month_name'] = df[date_column].dt.month
        df['Quarter_name'] = df[date_column].dt.quarter
        df = df.set_index(date_column)
        
        st.write(df)
        
        st.markdown("### Plot")
        st.line_chart(df[target_column])
        
        #st.markdown("### Correlation")
        #fig, ax = plt.subplots()
        #sn.heatmap(df.corr(), ax=ax,  annot=True)
        #st.write(fig)
        
        if ts_algorithm_name == 'VAR':
            split= st.sidebar.slider("Train/Test split %", 10, 90)
            
            split_value= int((split/100)*len(df))
            train_df= df.iloc[:split_value]
            test_df= df.iloc[split_value:]
            
            st.markdown("#### Length of Training set: "+ str(len(train_df)))
            st.markdown("#### Length of Test set: "+ str(len(test_df)))
            
            best_p, best_aic= run_auto_var(train_df, 12)
            model= VAR(train_df)
            model_fit= model.fit(best_p)
            
            st.markdown("#### Best P value: "+ str(best_p))
            
            prediction= model_fit.forecast(model_fit.y, steps=len(test_df))
            var_prediction= pd.DataFrame(prediction, columns=['Prediction', 'Var1', 'Var2', 'Var3'])
            var_prediction= var_prediction['Prediction'].values
            
            test_df['Prediction']= var_prediction
            display_results(train_df, test_df, target_column)
            
        
        
        
        elif ts_algorithm_name == 'K Nearest Neighbor':
            split= st.sidebar.slider("Train/Test split %", 10, 90)
            
            split_value= int((split/100)*len(df))
            train_df= df.iloc[:split_value]
            test_df= df.iloc[split_value:]
            
            st.markdown("#### Length of Training set: "+ str(len(train_df)))
            st.markdown("#### Length of Test set: "+ str(len(test_df)))
            
            best_k = get_best_k(train_df, test_df, target_column)
            st.markdown("#### Best K value: "+ str(best_k))
            
            knn= KNeighborsRegressor(n_neighbors= best_k)
            knn.fit(train_df.drop(target_column, axis=1), train_df[target_column])
            
            knn_prediction = knn.predict(test_df.drop(target_column, axis=1))
            test_df['Prediction']= knn_prediction
            
            display_results(train_df, test_df, target_column)
            
        elif ts_algorithm_name == 'Random Forest':
            split= st.sidebar.slider("Train/Test split %", 10, 90)
            run_default= st.sidebar.radio("Use default settings?",('Yes', 'No') )
            
            split_value= int((split/100)*len(df))
            train_df= df.iloc[:split_value]
            test_df= df.iloc[split_value:]
            
            st.markdown("#### Length of Training set: "+ str(len(train_df)))
            st.markdown("#### Length of Test set: "+ str(len(test_df)))
            
            if run_default == 'Yes':
                rf_model= RandomForestRegressor()
                     
            else:
                choose_n_estimators= st.sidebar.slider("No. of trees", 10, 500)
                choose_max_features= st.sidebar.selectbox("Max features", ['auto', 'sqrt', 0.2])
                choose_min_sample_leaf = st.sidebar.slider("Min Sample Leaf",10, 100)
                choose_oob_score= st.sidebar.selectbox("OOB Score", [True, False])
                
                rf_model= RandomForestRegressor(n_estimators=choose_n_estimators,
                                                max_features=choose_max_features,
                                                min_samples_leaf=choose_min_sample_leaf,
                                                oob_score=choose_oob_score)
                
            
            rf_model.fit(train_df.drop(target_column, axis=1), train_df[target_column])
            rf_prediction = rf_model.predict(test_df.drop(target_column, axis=1))
            test_df['Prediction']= rf_prediction
            display_results(train_df, test_df, target_column)
            
            
            
        
        else:
            st.info("Work In Progress")
            








