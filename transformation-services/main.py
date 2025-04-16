from fastapi import FastAPI

from util import TransformRequest, TransformResponse, load_dataset
from drain.manager import MultiDrainManager
from presidio.manager import MultiPresidioManager

presidio_manager = MultiPresidioManager("presidio/config/default.config")
drain_manager = MultiDrainManager("drain/config/default.config")

df = load_dataset("doc/dataset.csv") # Testing Dataset

app = FastAPI()

@app.get("/test")
def test_api():
    result = df.sample()
    return {"msg": result["msg"].iloc[0]}

@app.post("/full-transform")
def zero_example(req: TransformRequest):
    pii = presidio_manager.run(req.pii_detection_LANG, req.msg)
    template = drain_manager.run(req.template_miner_ID, req.msg, req.template_miner_LEARN)

    result = TransformResponse(
        template=template,
        hasPII=(len(pii) > 0)
    )

    return result
