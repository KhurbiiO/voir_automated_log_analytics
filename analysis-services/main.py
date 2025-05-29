from fastapi import FastAPI
import datetime

from util._pydantic import *
from util import DBClient
from metric.manager import MultiDogManager
from log.manager import MultiLBManager
import json

MONGO_URI = "mongodb://localhost:27017"
MONGO_DB = "solution_test"
MONGO_COLLECTION_1 = "test_1"
MONGO_COLLECTION_2 = "test_2"

client = DBClient(MONGO_URI, MONGO_DB)

dog_manager = MultiDogManager("metric/config/metric_anomaly.config")
bert_manager = MultiLBManager("log/config/log_anomaly.config")

app = FastAPI()

@app.get("/")
def first_example():
    return {"Test": "Analysis"}

@app.get("/load_log")
def load_log_database():
    return client.read_all(MONGO_COLLECTION_1).to_json(orient='records')

@app.get("/load_metric")
def load_metric_database():
    return client.read_all(MONGO_COLLECTION_2).to_json(orient='records')

@app.post("/metric")
def metric_check(req: DogSniffRequest):
    score = dog_manager.run(req.ID, req.value)

    result = DogSniffResponse(score>req.score_thresshold)

    return result

@app.post("/log")
def log_window_check(req: BERTWinRequest):
    df = client.read_window(datetime.datetime(req.start), datetime.datetime(req.end))
    templates = df["Template"].to_list()
    templates = [f"E{t}" for t in templates]

    sequences = []
    while True:
        if len(templates) > req.window:
            sequences.append(templates)
            break
        else:
            sequences.append(templates[:req.window])
            templates = templates[req.window:]
    
    result = [bert_manager.run(req.ID, " ".join(seq)) for seq in sequences]
    
    return json.dumps(result)