from fastapi import FastAPI

from util._pydantic import *
from util import load_dataset
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

@app.post("/smart-filter")
def get_smart_filter(req: SmartFilterRequest):
    anomaly, _ = drain_manager.run(req.template_miner_ID, req.msg, req.smart_filter_threshold)
    result = SmartFilterResponse(
        anomaly=anomaly
    )
    return result

@app.post("/pii-detection")
def get_pii_detection(req: PIIRequest):
    pii = presidio_manager.run(req.pii_detection_lang, req.msg)
    result = PIIResponse(
        PII=("True" if len(pii) > 0 else "False")
    )
    return result

@app.post("/template")
def get_template_miner(req: TemplateRequest):
    template = drain_manager.filters[req.template_miner_ID].get_template(req.cluster_ID)
    result = TemplateResponse(
        template=template
    )
    return result

@app.post("/full-transform")
def get_full_transform(req: TransformRequest):
    pii = presidio_manager.run(req.pii_detection_lang, req.msg)
    anomaly, clusterid = drain_manager.run(req.template_miner_ID, req.msg, req.smart_filter_threshold)

    result = TransformResponse(
        template_ID=clusterid,
        anomaly=anomaly,
        PII=("True" if len(pii) > 0 else "False")
    )

    return result
