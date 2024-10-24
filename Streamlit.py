import math
import pandas as pd
import pandas
import numpy as np
import streamlit as st
# Information
st.sidebar.markdown("## Authors' information")
st.sidebar.markdown("Author: Tan-Duy Phan")
st.sidebar.caption("Falculty of Civil Engineering-HCMUT-VNU\nEmail:\nptduy.sdh212@hcmut.edu.vn\nGoogle scholar:\nhttps://scholar.google.com/citations?user=Zi9MR6QAAAAJ&hl=en ")

#Tên bài báo
st.title ("Predicting the compressive strength of ultra high-performance concrete: an ensemble machine learning approach and actual application") 
# Chèn sơ đồ nghiên cứu
st.header("1. Layout of this study")
check1=st.checkbox('1.1 Display layout of this investigation')
if check1:
   st.image("Fig. 1.jpg", caption="Layout of this study")
# Hiển thị dữ liệu
st.header("2. Dataset")
check2=st.checkbox('2.1 Display dataset')
if check2:
   Database="Dataset.csv"
   df = pandas.read_csv(Database)
   df.head()
   st.write(df)
st.header("3. Modeling approach")
check3=st.checkbox('3.1 Display structure of Random Forest model')
if check3:
   st.image("Fig2.jpg", caption="Overview on structure of Random Forest model") 
#Make a prediction
st.header("4. Predicting compressive strength of UHPC using the RF model")
st.subheader("Input variables")
col1, col2, col3, col4 =st.columns(4)
with col1:
   X1=st.slider("Cement (kg/m\u00B3)", 270, 1250)
   X2 = st.slider("Silica fume (kg/m\u00B3)", 0, 375)
   X3= st.slider("Slag (kg/m\u00B3)", 0, 434)
   X4 = st.slider("Limestone powder (kg/m\u00B3)", 0, 1059)	
with col2:
   X5 =st.slider("Quartz powder (kg/m\u00B3)",0, 397)
   X6 =st.slider("Fly ash (kg/m\u00B3)",0, 356)
   X7=st.slider("Nano silica (kg/m\u00B3)", 0,48)
   X8=st.slider("Water (kg/m\u00B3)", 90,273)
with col3:
   X9=st.slider("Sand (kg/m\u00B3)", 0,1503)
   X10=st.slider("Coarse aggregate (kg/m\u00B3)", 0,1195)
   X11=st.slider("Fiber (kg/m\u00B3)", 0,234)
   X12=st.slider("Superplasticizer (kg/m\u00B3)", 1,57)	
with col4:	
   X13=st.slider("Relative humidity (%)", 50,100)
   X14=st.slider("Temperature (\u00B0C)", 20,210)
   X15=st.slider("Age (day)", 1,365)
Data="Dataset.csv"
df = pandas.read_csv(Data)
df.head()
X = df.iloc[:, 0:15].values 
y = df.iloc[:, 15].values 
#  model
from sklearn.ensemble import RandomForestRegressor
RF= RandomForestRegressor(max_depth=None,
	                       max_leaf_nodes=None,
	                       max_samples=None,
	                       min_samples_leaf=1,
	                       min_samples_split=2,
	                       min_impurity_decrease=0,
	                       n_estimators=1000,
	                       random_state=0)
RF.fit(X, y)
Inputdata = [X1, X2, X3, X4, X5, X6, X7,X8, X9, X10, X11, X12, X13, X14, X15]
from numpy import asarray
Newdata = asarray([Inputdata])
print(Newdata)
fc_pred2 = RF.predict(Newdata)
st.subheader("Output variable")
if st.button("Predict"):
    import streamlit as st
    import time
    progress_text = "Operation in progress. Please wait."
    my_bar = st.progress(0, text=progress_text)
    for percent_complete in range(100):
       time.sleep(0.01)
       my_bar.progress(percent_complete + 1, text=progress_text)
    time.sleep(1)
    my_bar.empty()
    st.success(f"Your predicted compressive strength (MPa) of UHPC from  RF model is {(fc_pred2)}")

