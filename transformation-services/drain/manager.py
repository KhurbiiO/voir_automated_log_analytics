from util import load_config, Vectorizer

from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
from drain3.file_persistence import FilePersistence
from drain import SmartFilter

class MultiSMManager:
    def __init__(self, filepath: str):
        self.config = load_config(filepath)
        self.filters = {}
        self.vectorizer = Vectorizer() #init vectorizer

        for key in self.config.keys():
            drain_config = TemplateMinerConfig()
            drain_config.load(self.config[key]["config"])
            persistence = FilePersistence(self.config[key]["state"])
            miner = TemplateMiner(persistence)
            self.filters[key] = SmartFilter(miner, self.vectorizer)

    def get_filter (self, key):
        return self.filters[key]
    
    def get_cluster_ID(self, key, msg, learn):
        self.filters[key].get_cluster_ID(msg, learn)

    def get_template(self, key, id):
        self.filters[key].get_template(id)

    def run(self, key, msg, tau):
        return self.filters[key].inference(msg, tau)


