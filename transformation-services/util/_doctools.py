import os
import yaml
import pandas as pd

def append(directory: str, filepath:str):
    if os.path.isdir(directory):
        return os.path.join(directory, filepath)
    return None      

def load_config(filepath: str) -> dict:
    with open(filepath, "r") as file:
        return yaml.safe_load(file)
    return None

def load_dataset(filepath: str):
    return pd.read_csv(filepath)
