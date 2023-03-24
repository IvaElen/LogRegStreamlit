import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as ex
from sklearn import preprocessing

class LogReg:
    def __init__(self, learning_rate=0.1):
        self.learning_rate = learning_rate
        
    def fit(self, X, y):
        
        X = np.array(X)
        y = np.array(y)
        self.coef_ = np.random.rand(X.shape[1])
        self.intercept_ = np.random.rand(1)
        
        n_epoch = 2000
        for epoch in range(n_epoch):
            
            z = self.intercept_+X@self.coef_
            yhat = 1/(1+np.exp(-z))
            
            grad = X.T*(yhat-y)
            grad_0 = yhat-y
            
            self.coef_ -= self.learning_rate*(grad.mean(axis=1))
            self.intercept_ -= self.learning_rate*(grad_0.mean())
            


st.write("""
## Привет! Я здесь, чтобы помочь тебе рассчитать веса методом логистической регрессии для твоего датафрейма!

### Загрузи свой CSV-файл и выбери столбец, в котором расположены таргентные значения
""")
uploaded_file = st.file_uploader("Загрузите свой CSV файл",type = ['csv'])

if uploaded_file is not None:
    in_df = pd.read_csv(uploaded_file).drop('Unnamed: 0', axis=1)
    drop = st.selectbox(label = 'Выбери столбец с таргетом', options = in_df.columns)
    scaler = preprocessing.StandardScaler()
    names = in_df.drop(columns=drop).columns
    d = scaler.fit_transform(in_df.drop(columns=drop))
    X = pd.DataFrame(d, columns=names)
    y = in_df[drop]
    lg = LogReg()
    lg.fit(X, y)
    data = pd.DataFrame(dict(zip(('w%d' %i for i in range(1, 1+len(X.columns))), np.round(lg.coef_, 3))),index = range(1))
    data['w0']=np.round(lg.intercept_,3)
    st.dataframe(data)
    
    st.write("""
    ### Бонус!:)
    """)
    x = st.selectbox(label = 'Выбери фичу по оси x', options = names.insert(0,''))
    y = st.selectbox(label = 'Выбери фичу по оси y', options = names.insert(0,''))

    if x!= '' and y!='':
        fig1 = ex.scatter(x = in_df[x], y = in_df[y])
        fig2 = ex.bar(x =in_df[x], y = in_df[y])
        st.plotly_chart(fig1)
        st.plotly_chart(fig2)
    else:
        st.plotly_chart(ex.scatter(in_df))
        st.plotly_chart(ex.bar(in_df))
