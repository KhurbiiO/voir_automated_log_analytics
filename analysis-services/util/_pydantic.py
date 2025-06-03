from pydantic import BaseModel

class DogSniffPreload(BaseModel):
    ID : str
    start: str
    end: str


class DogSniffRequest(BaseModel):
    ID : str
    score_thresshold : float
    value: int

class DogSniffResponse (BaseModel):
    is_Anonamly : bool

class WinRequest(BaseModel):
    ID: str
    start: str
    end: str
    window: int

class WinResponse(BaseModel):
    result: str


