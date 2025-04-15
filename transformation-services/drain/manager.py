from util import load_config

from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
from drain3.file_persistence import FilePersistence

class MultiDrainManager:
    def __init__(self, filepath: str):
        self.config = load_config(filepath)
        self.states = {}

        for key in self.config.keys():
            drain_config = TemplateMinerConfig()
            drain_config.load(self.config[key]["CONFIG"])
            persistence = FilePersistence(self.config[key]["STATE"])
            miner = TemplateMiner(persistence)
            self.states.update({key: miner})
            
    def run(self, key: str, msg: str, learn: bool):
        if learn:
            return self.states[key].add_log_message(msg)["template_mined"]
        else:
            cluster = self.states[key].match(msg)
            return cluster.get_template()


