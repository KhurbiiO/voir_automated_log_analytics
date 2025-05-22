from pydantic import BaseModel

class TransformRequest(BaseModel):
    template_miner_ID: str
    template_miner_LEARN : bool
    pii_detection_LANG: str
    msg: str

class TransformResponse (BaseModel):
    template: str
    hasPII: str




