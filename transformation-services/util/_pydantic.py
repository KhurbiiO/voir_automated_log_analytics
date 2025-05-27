from pydantic import BaseModel

class TransformRequest(BaseModel):
    template_miner_ID: str
    smart_filter_THRESHOLD : float
    pii_detection_LANG: str
    msg: str

class TransformResponse (BaseModel):
    template: int
    anomaly: bool
    PII: bool




