
import numpy as np
import pandas as pd

from numpy.linalg import norm
from drain3 import TemplateMiner
from util import Vectorizer

class SmartFilter():
    def __init__(self, miner:TemplateMiner, vectorizer:Vectorizer):
        self.whitelist = []
        self.miner = miner
        self.vectorizer = vectorizer

    def get_template(self, id):
        return self.miner.drain.id_to_cluster[id].get_template()

    def get_cluster_ID(self, msg, learn):
        if learn:
            self.miner.add_log_message(msg)["cluster_id"]
        else:
            cluster = self.miner.match(msg)
            if cluster:
                return cluster.cluster_id
            else:
                return -1

    def inference(self, msg, tau=0.5):
        cluster = self.miner.match(msg)
        # Unique messages / unknown in terms of structure or content
        if cluster is None:
            add = self.miner.add_log_message(msg)
            self.whitelist.insert(0, (add["cluster_id"], msg))
            return True, add["cluster_id"]
        
        if cluster is not None:
            # Check if in whitelist
            if cluster.cluster_id in [entry[0] for entry in self.whitelist]:            
                template = cluster.get_template()
                params = self.miner.extract_parameters(template, msg)

                index = [i for i in range(len(self.whitelist)) if self.whitelist[i][0] == cluster.cluster_id] # Find where IDs are equal

                # Check for content match with whitelist
                for i in index:
                    params_compare = self.miner.extract_parameters(template, self.whitelist[i][1])
                    if params and params_compare:
                        simularities = [self.calc_cosine_simalirity(
                            self.vectorizer.get_vector(params[i].value),
                            self.vectorizer.get_vector(params_compare[i].value)
                            ) 
                            for i in range(len(params))
                        ]
                        average_simalirity = sum(simularities)/len(simularities)
                        if average_simalirity >= tau:
                            return True, cluster.cluster_id
        return False, cluster.cluster_id
    
    def calc_cosine_simalirity(A, B):
        return np.dot(A,B)/(norm(A)*norm(B))

    def create_template_whitelist(self, df: pd.DataFrame):
        result = []
        for _, line in df.iterrows():
            msg = line["_value"]
            cluster = self.miner.match(msg)
            if cluster:
                result.append((cluster.cluster_id, msg))
            else:
                # Update miner with new messages
                add = self.miner.add_log_message(msg)
                result.append((add["cluster_id"], msg))

        self.whitelist = result
                