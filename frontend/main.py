import streamlit as st
import requests
import pandas as pd
import json

st.title("VOIR - Automated Support Engineering")

# Make HTTP request
try:
    st.header("Log Analytics")
    response = requests.get("http://localhost:8000/load_log")
    response.raise_for_status()
    data = response.json()

    # Convert to DataFrame
    df = pd.DataFrame(data)

    st.success("Data loaded from API")
    st.dataframe(df)

    st.divider()

    start = st.date_input("Start-date:")
    end = st.date_input("End-date:")
    seq_len = st.number_input("Sequence Analysis Length:", min_value=1, step=1)

    button = st.button("Run LogBERT Analysis")
    button1 = st.button("Run DeepLog Analysis")

    if button:
        req = {
            "ID" : "fine",
            "start" : start.isoformat(),
            "end" : end.isoformat(),
            "window" : seq_len
        }

        response = requests.post("http://localhost:8000/logbert", json=req)

        if response.status_code == 200:
            data = response.json()

            st.text(data)
    
    elif button1:
        req = {
            "ID" : "test2",
            "start" : start.isoformat(),
            "end" : end.isoformat(),
            "window" : seq_len
        }

        response = requests.post("http://localhost:8000/deeplog", json=req)

        if response.status_code == 200:
            data = response.json()

            st.text(data)

    st.divider()

    button2 = st.button("Filter data")

    if button2:
        data = df[df["SmartFilter"] == "true"]
        st.dataframe(data)


    st.divider()


except requests.exceptions.RequestException as e:
    st.error(f"Failed to fetch data: {e}")


# Make HTTP request
try:
    st.header("Metric Analytics")
    response = requests.get("http://localhost:8000/load_metric")
    response.raise_for_status()
    data = response.json()

    # Convert to DataFrame
    df = pd.DataFrame(data)

    st.success("Data loaded from API")

    st.line_chart(
        data=df.set_index("@timestamp")["Value"],
        use_container_width=True
    )

    start = st.date_input("Metric Train Start-date:")
    end = st.date_input("Metric Train End-date:")

    button = st.button("Train Dog")

    if button:
        req = {
            "ID" : "test",
            "start" : start.isoformat(),
            "end" : end.isoformat(),
        }

        requests.post("http://localhost:8000/metric_load_db", json=req)

        if response.status_code == 200:
            data = response.json()

            st.text(data)

    
    metric_input = st.number_input("Metric Input:")
    
    if metric_input:
        req = {
            "ID" : "test",
            "score_thresshold" : 0.05,
            "value" : int(metric_input),
        }

        response = requests.post("http://localhost:8000/metric", json=req)

        if response.status_code == 200:
            data = response.json()

            st.text(data)

except requests.exceptions.RequestException as e:
    st.error(f"Failed to fetch data: {e}")
