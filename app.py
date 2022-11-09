import pickle
import requests
from PIL import Image

import pandas as pd
import streamlit as st
import plotly.express as px
from fbprophet import Prophet

im = Image.open('image/favicon.ico')

st.set_page_config(
    page_title='Persistent Disk Storage Forecasting App', 
    page_icon = im, 
    layout = 'wide', 
    initial_sidebar_state = 'auto'
)

st.sidebar.image('image/logo.png')
st.sidebar.title('Persistent Disk Storage Forecasting')
st.sidebar.write('---')

st.write("""
# Persistent Disk Storage Forecasting

Persistent Disk Storage Forecasting **provides 24 hours future forecast of available disk storage using historical data.**
""")
st.write('---')

# Loads Dataset
data_path = 'data/sample.csv'
data_df = pd.read_csv(data_path, index_col = [0])
data_df = data_df.T.copy().reset_index().rename(mapper={'index':'date','product_1':'product_1'}, axis=1)
st.write(data_df.head(20))
data_df = data_df.rename(mapper={'date':'ds','product_1':'y'}, axis=1)

# showing fig1
st.header(' Disk Storage with Date')
fig = px.line(data_df, x='ds', y='y')
fig.update_yaxes(title_text = ' Disk Storage')
fig.update_xaxes(title_text='Date')
fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide', margin=dict(t=0, b=0, l=0, r=0))
st.plotly_chart(fig, use_container_width=False)

# Sidebar
# Header of Specify Input Parameters
st.sidebar.header('Specify Input Parameters')

def user_input_features():
    
    return st.sidebar.slider('Period', 1, 40, 30)

# Main Panel

# to retrain
agree = st.checkbox('Check to retrain the model')
filename = 'model/finalized_model.sav'
if agree:

    # Build Regression Model
    model = Prophet()
    model.fit(data_df)

    # save the model to disk
    pickle.dump(model, open(filename, 'wb'))
else:
    # load the model from disk
    model = pickle.load(open(filename, 'rb'))

period = user_input_features()

# Print specified input parameters
st.header('Specified Input parameters')
st.write(f'Period: **{period}**')
st.write('---')

# Apply Model to Make Prediction
if st.sidebar.button('Prediction'):
    future = model.make_future_dataframe(periods=period)
    forecast = model.predict(future)
    st.header('Persistent Disk Storage Forecast')
    st.write('Forecasting Scores:')
    st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])
    st.write('---')

    st.header('Forecasted Dataframe Plot')
    st.write(model.plot(forecast))
    st.header('Forecasted Components Plot')
    st.write(model.plot_components(forecast))
else:
    st.warning('Please Click on Prediction')
st.write('---')
