from pydantic import BaseModel

class TransformRequest(BaseModel):
    template_miner_ID: str
    smart_filter_threshold : float
    pii_detection_lang: str
    msg: str

class SmartFilterRequest(BaseModel):
    template_miner_ID: str
    smart_filter_threshold: float
    msg: str

class TemplateRequest(BaseModel):
    template_miner_ID: str
    cluster_ID: int

class PIIRequest(BaseModel):
    pii_detection_lang: str
    msg: str

class TransformResponse (BaseModel):
    template_ID: int
    anomaly: bool
    PII: bool

class SmartFilterResponse(BaseModel):
    anomaly: bool

class TemplateResponse(BaseModel):
    template: str

class PIIResponse(BaseModel):
    PII: bool




