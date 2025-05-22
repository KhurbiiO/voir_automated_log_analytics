from fastapi import FastAPI

from util import TransformRequest, TransformResponse, load_dataset
from drain.manager import MultiDrainManager
from presidio.manager import MultiPresidioManager

presidio_manager = MultiPresidioManager("presidio/config/default.config")
drain_manager = MultiDrainManager("drain/config/default.config")

df = load_dataset("doc/dataset.csv") # Testing Dataset

app = FastAPI()

@app.get("/test")
def test():
    result = df.sample()
    return {"msg": result["msg"].iloc[0]}

@app.get("/get-template")
def get_template():
    pass

@app.post("/full-transform")
def get_full_transform(req: TransformRequest):
    pii = presidio_manager.run(req.pii_detection_LANG, req.msg)
    template = drain_manager.run(req.template_miner_ID, req.msg, req.template_miner_LEARN)

    result = TransformResponse(
        template=template,
        hasPII=("True" if len(pii) > 0 else "False")
    )

    return result
