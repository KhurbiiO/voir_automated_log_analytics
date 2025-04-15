import os
import yaml

def append(directory: str, filepath:str):
    if os.path.isdir(directory):
        return os.path.join(directory, filepath)
    return None      

def load_config(filepath: str) -> dict:
    with open(filepath, "r") as file:
        return yaml.safe_load(file)
    return None

