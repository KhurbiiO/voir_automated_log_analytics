import pandas as pd
from pymongo import MongoClient

class DBClient():
    def __init__(self, uri, db):
        pass
        # Configuration
        self.MONGO_URI = uri
        self.MONGO_DB = db

        # Connect to MongoDB
        self.client = MongoClient(self.MONGO_URI)
        self.db = self.client[self.MONGO_DB]
        # self.collection = self.db[self.MONGO_COLLECTION]

    # Write Benchmark
    # def benchmark_write(self, data):
    #     self.collection.insert_many(data)

    # Read Benchmark
    def read_window(self, start, end, collection):
        collection = self.db[collection]
        records = collection.find({"timestamp": {"$gte": start, "$lte": end}})
        return self.to_df(records)
    
    def read_all(self, collection):
        collection = self.db[collection]
        records = collection.find()
        return self.to_df(records)

    def to_df(self, data):
        data = list(data)  # Convert cursor to a list
        df = pd.DataFrame(data)
        return df