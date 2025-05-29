from pydantic import BaseModel

class DogSniffRequest(BaseModel):
    ID : str
    score_thresshold : float
    value: int

class DogSniffResponse (BaseModel):
    is_Anonamly : bool

class BERTWinRequest(BaseModel):
    ID: str
    start: str
    end: str
    window: int

class BERTWinResponse(BaseModel):
    result: str


