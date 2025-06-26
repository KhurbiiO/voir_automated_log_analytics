from fastapi import FastAPI
from datetime import datetime
import pandas as pd

from util._pydantic import *
from util import DBClient
from metric.manager import MultiMAManager
from log.manager import MultiLAManager
import json

from fastapi.responses import JSONResponse

MONGO_URI = "mongodb://mongo:27017"
MONGO_DB = "solution_test"
MONGO_COLLECTION_1 = "test_1"
MONGO_COLLECTION_2 = "test_2"

client = DBClient(MONGO_URI, MONGO_DB)

ma_manager = MultiMAManager("metric/config/metric_anomaly.config")
la_manager = MultiLAManager("log/config/log_anomaly.config")

app = FastAPI()

@app.get("/load_log")
def load_log_database():
    data = client.read_all(MONGO_COLLECTION_1)

    return JSONResponse(content=list(data))

@app.get("/load_metric")
def load_metric_database():
    data = client.read_all(MONGO_COLLECTION_2)
    
    return JSONResponse(content=list(data))

@app.post("/metric_preload")
def load_metric_db(req:MetricPreload):
    df = pd.DataFrame(client.read_window(datetime.fromisoformat(req.start), datetime.fromisoformat(req.end), MONGO_COLLECTION_2))
    values = df["Value"]
    try:
        ma_manager.analyzers[req.ID].train(values)
    except ValueError:
        return JSONResponse(content={"status": "Error", "message": "Training failed. Check the data."}, status_code=400)
    return JSONResponse(content={"status": "Success", "message": "Training completed successfully."})


@app.post("/metric")
def metric_check(req: MetricAnalysisRequest):
    score = ma_manager.run(req.ID, req.value)

    result = MetricAnalysisResponse(anonamly=score>req.score_thresshold)

    return result

@app.post("/logbert")
def log_window_check(req: LogAnalysisRequest):
    df = pd.DataFrame(client.read_window(datetime.fromisoformat(req.start), datetime.fromisoformat(req.end), MONGO_COLLECTION_1))
    templates = df["Template"].to_list()
    templates = [f"E{t}" for t in templates]

    sequences = []
    for i in range(0, len(templates), req.window):
        sequences.append(templates[i:i+req.window])
    
    result = LogAnalysisResponse(
        result=json.dumps([la_manager.run(req.ID, " ".join(seq))["anomaly"] for seq in sequences])
    )
    
    return result

@app.post("/deeplog")
def log_window_check(req: LogAnalysisRequest):
    df = pd.DataFrame(client.read_window(datetime.fromisoformat(req.start), datetime.fromisoformat(req.end), MONGO_COLLECTION_1))
    sequences = []
    for i in range(0, len(df), req.window):
        chunk = df.iloc[i:i + req.window].reset_index(drop=True)
        sequences.append(chunk)
    
    result = LogAnalysisResponse(
        result=json.dumps([la_manager.run(req.ID, seq)[0] for seq in sequences])
    )    
    return result