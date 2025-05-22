from fastapi import FastAPI

from util import DogSniffRequest, DogSniffResponse
from metric.manager import MultiDogManager

dog_manager = MultiDogManager("metric/config/metric_anomaly.config")

app = FastAPI()

@app.get("/")
def first_example():
    return {"Test": "Analysis"}

@app.post("/full-transform")
def get_full_transform(req: DogSniffRequest):
    score = dog_manager.run(req.dog_ID, req.value)

    result = DogSniffResponse(score>req.dog_score_thresshold)

    return result