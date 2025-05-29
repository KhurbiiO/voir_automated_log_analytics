import sys
sys.path.append("log")

from bert_pytorch import Predictor

from util import load_config

class MultiLBManager:
    def __init__(self, filepath: str):
        self.config = load_config(filepath)
        self.logberts = {}

        for key in self.config.keys():
            directory = self.config[key]["dir"]
            model_directory = self.config[key]["model_dir"]
            seq_len = self.config[key]["seq_len"]

            options = {}
            options['output_dir'] = directory
            options["model_dir"] = model_directory
            options["model_path"] = options["model_dir"] + "best_bert.pth"
            options["vocab_path"] = directory + "/vocab.pkl"
            options["seq_len"] = seq_len

            self.logberts[key] = Predictor(options)

    def run(self, key:str, val):
        logbert = self.logberts[key]
        return logbert.predict_single_sequence(val)
    
