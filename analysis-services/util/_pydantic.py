from pydantic import BaseModel

class DogSniffRequest(BaseModel):
    dog_score_thresshold : float
    dog_ID : str
    value: int

class DogSniffResponse (BaseModel):
    is_Anonamly : bool




