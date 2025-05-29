import streamlit as st
import pandas as pd

st.title("VOIR - Automated Support Engineering")

load_all = st.button("Load Data")
if load_all:
    st.write("### Full Dataset")
    st.dataframe(client.read_all())

st.divider()

st.header("Analysis Service")

start = st.date_input("Start")
end = st.date_input("End")

load_window = st.button("Load Window")
df = None
if load_window:
    st.write("### Window Dataset")
    df = client.read_window(start.isoformat(), end.isoformat())
    st.dataframe(df)