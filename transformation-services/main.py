from fastapi import FastAPI
from drain.manager import MultiDrainManager
from presidio.manager import MultiPresidioManager

presidio_manager = MultiPresidioManager("presidio/config/default.config")
drain_manager = MultiDrainManager("drain/config/default.config")

app = FastAPI()

@app.get("/pii")
def zero_example():
    return {"Test": "Tranformation"}

@app.get("/pii")
def first_example():
    return {"Test": "Tranformation"}

@app.get("/template")
def second_example():
    return {"Test": "Tranformation"}