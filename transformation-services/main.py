from fastapi import FastAPI

from util import TransformRequest, TransformResponse, load_dataset
from drain import MultiSMManager
from presidio import MultiPresidioManager

presidio_manager = MultiPresidioManager("presidio/config/manager.config")
drain_manager = MultiSMManager("drain/config/manager.config")

log_df = load_dataset("doc/log_sim.csv")  
metric_df = load_dataset("doc/metric_sim.csv")

app = FastAPI()

app.state.log_count = 0
app.state.metric_count = 0

@app.get("/test1")
def test1():
    i = app.state.log_count
    app.state.log_count += 1
    result = log_df["_value"][i % len(log_df)]
    return {"msg": result}

@app.get("/test2")
def test2():
    i = app.state.metric_count
    app.state.metric_count += 1
    result = metric_df["_value"][i % len(metric_df)]
    return {"msg": float(result)}

@app.post("/full-transform")
def get_full_transform(req: TransformRequest):
    pii = presidio_manager.run(req.pii_detection_LANG, req.msg)
    anomaly, clusterid = drain_manager.run(req.template_miner_ID, req.msg, req.smart_filter_THRESHOLD)

    result = TransformResponse(
        template=clusterid,
        anomaly=anomaly,
        PII=("True" if len(pii) > 0 else "False")
    )

    return result
