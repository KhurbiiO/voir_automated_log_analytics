import streamlit as st
import requests
import pandas as pd
import json
from datetime import datetime

st.set_page_config(page_title="VOIR - Log Analytics", layout="wide")
st.title("🔍 VOIR - Automated Support Engineering")

# Load logs
st.header("📑 Log Analytics")

try:
    response = requests.get("http://localhost:8000/load_log")
    response.raise_for_status()
    data = response.json()

    df = pd.DataFrame(data)
    df["@timestamp"] = pd.to_datetime(df["@timestamp"], errors='coerce')
    st.success("✅ Log data successfully loaded from API.")
    with st.expander("📊 View Raw Log Data"):
        st.dataframe(df)

    st.divider()

    # LogBERT and DeepLog Analysis
    st.subheader("🧠 Log Anomaly Detection")

    col1, col2, col3 = st.columns(3)
    with col1:
        start = st.date_input("Start date:")
    with col2:
        end = st.date_input("End date:")
    with col3:
        seq_len = st.number_input("Sequence Length", min_value=1, step=1, value=10)

    col4, col5 = st.columns(2)
    run_logbert = col4.button("🚀 Run LogBERT Analysis")
    run_deeplog = col5.button("🚀 Run DeepLog Analysis")

    if run_logbert:
        with st.spinner("Running LogBERT..."):
            try:
                req = {"ID": "fine", "start": start.isoformat(), "end": end.isoformat(), "window": seq_len}
                response = requests.post("http://localhost:8000/logbert", json=req)
                response.raise_for_status()
                st.success("LogBERT Results")
                st.json(response.json()["result"])
            except Exception as e:
                st.error(f"LogBERT request failed: {e}")

    if run_deeplog:
        with st.spinner("Running DeepLog..."):
            try:
                req = {"ID": "test2", "start": start.isoformat(), "end": end.isoformat(), "window": seq_len}
                response = requests.post("http://localhost:8000/deeplog", json=req)
                response.raise_for_status()

                anomaly_scores = json.loads(response.json()["result"])
                filtered_df = df[(df["@timestamp"] >= start.isoformat()) & (df["@timestamp"] <= end.isoformat())]

                sequences = [filtered_df.iloc[i:i + seq_len].reset_index(drop=True)
                             for i in range(0, len(filtered_df), seq_len)]
                assert len(sequences) == len(anomaly_scores)

                anomalies = [(i, score, seq) for i, (score, seq) in enumerate(zip(anomaly_scores, sequences)) if score < 0.5]

                for i, score, seq in anomalies:
                    st.divider()
                    st.text(f"🔴 Anomaly Detected (Score: {1 - score:.2f}) in Chunk #{i}")
                    st.dataframe(seq)

                st.divider()
                st.success(f"✅ Anomalies Found: {len(anomalies)} / {len(anomaly_scores)} chunks")

            except Exception as e:
                st.error(f"DeepLog request failed: {e}")

    # Smart filter
    with st.expander("🧹 Smart Filter Logs"):
        if st.button("Apply SmartFilter"):
            if "SmartFilter" in df.columns:
                filtered = df[df["SmartFilter"] == "true"]
                st.dataframe(filtered)
            else:
                st.warning("No 'SmartFilter' column found in data.")

    st.divider()

    # Template Mining
    st.subheader("🧬 Template Cluster Analysis")

    clusterID = st.number_input("Template Cluster ID:", min_value=1, step=1)
    if st.button("🔍 Get Template"):
        with st.spinner("Fetching template..."):
            try:
                req = {"template_miner_ID": "test", "cluster_ID": clusterID}
                response = requests.post("http://localhost:8001/template", json=req)
                response.raise_for_status()
                st.success("Template fetched successfully.")
                st.json(response.json())
            except Exception as e:
                st.error(f"Failed to fetch template: {e}")

except requests.exceptions.RequestException as e:
    st.error(f"Failed to load log data from API: {e}")


st.header("📈 Metric Analytics")

try:
    response = requests.get("http://localhost:8000/load_metric")
    response.raise_for_status()
    data = response.json()

    df = pd.DataFrame(data)
    df["@timestamp"] = pd.to_datetime(df["@timestamp"], errors='coerce')
    df = df.set_index("@timestamp").sort_index()

    st.success("✅ Metric data successfully loaded.")
    
    with st.expander("📊 View Metric Time Series"):
        st.line_chart(data=df["Value"], use_container_width=True)

    st.divider()

    # Training section
    st.subheader("🎯 Train Metric Model")
    col1, col2 = st.columns(2)
    with col1:
        train_start = st.date_input("Training Start Date:")
    with col2:
        train_end = st.date_input("Training End Date:")

    if st.button("🚀 Train Model"):
        with st.spinner("Sending training data..."):
            try:
                train_req = {
                    "ID": "test",
                    "start": train_start.isoformat(),
                    "end": train_end.isoformat()
                }
                train_resp = requests.post("http://localhost:8000/metric_preload", json=train_req)
                train_resp.raise_for_status()
                st.success("Training request sent successfully.")
                st.json(train_resp.json())
            except Exception as e:
                st.error(f"Training failed: {e}")

    st.divider()

    # Prediction section
    st.subheader("🔍 Metric Prediction")
    metric_input = st.number_input("Enter Metric Value for Prediction:")

    if st.button("📌 Predict"):
        with st.spinner("Predicting..."):
            try:
                predict_req = {
                    "ID": "test",
                    "score_thresshold": 0.05,
                    "value": int(metric_input)
                }
                pred_resp = requests.post("http://localhost:8000/metric", json=predict_req)
                pred_resp.raise_for_status()
                st.success("Prediction Result")
                st.json(pred_resp.json())
            except Exception as e:
                st.error(f"Prediction failed: {e}")

except requests.exceptions.RequestException as e:
    st.error(f"❌ Failed to load metric data: {e}")