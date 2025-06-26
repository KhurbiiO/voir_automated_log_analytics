from pydantic import BaseModel

class MetricPreload(BaseModel):
    ID : str
    start: str
    end: str

class MetricAnalysisRequest(BaseModel):
    ID : str
    score_thresshold : float
    value: float

class MetricAnalysisResponse (BaseModel):
    anonamly : bool

class LogAnalysisRequest(BaseModel):
    ID: str
    start: str
    end: str
    window: int

class LogAnalysisResponse(BaseModel):
    result: str


