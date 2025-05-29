from pymongo import MongoClient
import datetime

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
        records = list(collection.find({"@timestamp": {"$gte": start, "$lte": end}}))
        self.clean_records(records)
        return records
    
    def read_all(self, collection):
        collection = self.db[collection]
        records = list(collection.find())
        self.clean_records(records)

        return records

    def clean_records(self, records):
        for doc in records:
            if '_id' in doc:
                doc['_id'] = str(doc['_id'])

            for k, v in doc.items():
                if isinstance(v, datetime.datetime):
                    doc[k] = v.isoformat()