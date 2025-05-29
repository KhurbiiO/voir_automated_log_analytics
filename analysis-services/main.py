from fastapi import FastAPI
from datetime import datetime
import pandas as pd

from util._pydantic import *
from util import DBClient
from metric.manager import MultiDogManager
from log.manager import MultiLBManager
import json

from fastapi.responses import JSONResponse

MONGO_URI = "mongodb://mongo:27017"
MONGO_DB = "solution_test"
MONGO_COLLECTION_1 = "test_1"
MONGO_COLLECTION_2 = "test_2"

client = DBClient(MONGO_URI, MONGO_DB)

dog_manager = MultiDogManager("metric/config/metric_anomaly.config")
bert_manager = MultiLBManager("log/config/log_anomaly.config")

app = FastAPI()

@app.get("/load_log")
def load_log_database():
    data = client.read_all(MONGO_COLLECTION_1)

    return JSONResponse(content=list(data))

@app.get("/load_metric")
def load_metric_database():
    data = client.read_all(MONGO_COLLECTION_2)
    
    return JSONResponse(content=list(data))

@app.post("/metric")
def metric_check(req: DogSniffRequest):
    score = dog_manager.run(req.ID, req.value)

    result = DogSniffResponse(is_Anonamly=score>req.score_thresshold)

    return result

@app.post("/log")
def log_window_check(req: BERTWinRequest):
    df = pd.DataFrame(client.read_window(datetime.fromisoformat(req.start), datetime.fromisoformat(req.end), MONGO_COLLECTION_1))
    templates = df["Template"].to_list()
    templates = [f"E{t}" for t in templates]

    sequences = []
    for i in range(0, len(templates), req.window):
        sequences.append(templates[i:i+req.window])
    
    result = [bert_manager.run(req.ID, " ".join(seq))["anomaly"] for seq in sequences]
    
    return json.dumps(result)