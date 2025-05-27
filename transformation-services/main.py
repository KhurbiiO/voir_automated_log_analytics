from fastapi import FastAPI

from util import TransformRequest, TransformResponse, load_dataset
from drain import MultiSMManager
from presidio import MultiPresidioManager

presidio_manager = MultiPresidioManager("presidio/config/manager.config")
drain_manager = MultiSMManager("drain/config/manager.config")

df = load_dataset("doc/dataset.csv") # Testing Dataset

app = FastAPI()

@app.get("/test")
def test():
    result = df.sample()
    return {"msg": result["msg"].iloc[0]}

@app.post("/full-transform")
def get_full_transform(req: TransformRequest):
    pii = presidio_manager.run(req.pii_detection_LANG, req.msg)
    anomaly, clusterid = drain_manager.run(req.template_miner_ID, req.msg, req.smart_filter_THRESHOLD)

    result = TransformResponse(
        template_ID=clusterid,
        anomaly=anomaly,
        hasPII=("True" if len(pii) > 0 else "False")
    )

    return result
