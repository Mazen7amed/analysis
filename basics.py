import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


# loading data
df = pd.read_csv('company_sales_data.csv')
st.write(df.head())

# matplotlib
st.header("Matplotlib")
st.subheader("Line Plot")
fig = plt.figure(figsize=(15,8))
plt.plot(df['month_number'], df['total_profit'], c='r', lw=3, marker='^', markersize=10, ls='--')
plt.title("Month vs. Profit", fontsize=20)
plt.xlabel("Months")
plt.ylabel('Profit')
st.pyplot(fig)
st.text("This is a text descripting the previous figure")

st.subheader("Scatter Plot")
fig = plt.figure(figsize=(15,8))
plt.scatter(df['month_number'], df['total_profit'])
plt.title("Month vs. Profit", fontsize=20)
plt.xlabel("Months")
plt.ylabel('Profit')
st.pyplot(fig)

st.subheader("Histogram")
fig = plt.figure(figsize=(15,8))
plt.hist(df['total_profit'], bins=10)
st.pyplot(fig)

st.subheader('Bar Chart')
fig = plt.figure(figsize=(15,8))
plt.bar(df['month_number'], df['total_profit'])
plt.title("Month vs. Profit", fontsize=20)
plt.xlabel("Months")
plt.ylabel('Profit')
st.pyplot(fig)

st.header('Streamlit')
st.subheader("Line Chart")
st.line_chart(data=df)

st.header("Seaborn")
df = sns.load_dataset('tips')
st.dataframe(df.head())

st.subheader('Histogram')
fig = plt.figure(figsize=(15,8))
sns.distplot(df['total_bill'])
st.pyplot(fig)

st.subheader("Scatter Plot")
option = st.selectbox("Select an option", ['sex','smoker','day','time'])
fig = plt.figure(figsize=(15,8))
sns.scatterplot(data=df, x=df['total_bill'], y=df['tip'], hue=option)
st.pyplot(fig)


st.subheader("Box Plot")
fig = plt.figure(figsize=(15,8))
options = st.radio("Select an option", ['total_bill','tip'])
sns.boxplot(data=df, y=options)
st.pyplot(fig)


st.subheader("Heatmap")
df2=df[['total_bill','tip']]
fig = plt.figure(figsize=(15,8))
sns.heatmap(df2.corr(), annot=True)
st.pyplot(fig)

st.header("Plotly")

st.subheader("Scatter Plot")
select = st.selectbox("Select an option", ['sex','smoker','day','time'], key='A')
fig = px.scatter(data_frame=df, x='total_bill', y='tip', color=select)
st.plotly_chart(fig)

st.subheader("Bar Chart")
fig = px.bar(data_frame=df, x=df['smoker'], color='sex')
st.plotly_chart(fig)
st.subheader("Histogram")
fig = px.histogram(df['tip'])
st.plotly_chart(fig)

st.subheader('Violin Plot')
fig = px.violin(df['total_bill'])
st.plotly_chart(fig)
st.subheader("Box plot")
fig = px.box(df, y='total_bill')
st.plotly_chart(fig)






