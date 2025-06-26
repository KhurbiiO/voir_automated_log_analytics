import sys
sys.path.append("log")

from bert_pytorch import Predictor
from log_analyzer import DeepLogAnalyzer, LogBERTAnalyzer

from util import load_config

class MultiLAManager:
    def __init__(self, filepath: str):
        self.config = load_config(filepath)
        self.log_analyzers = {}

        for key in self.config.keys():
            match self.config[key]["type"]:
                case "LB":
                    directory = self.config[key]["dir"]
                    model_directory = self.config[key]["model_dir"]
                    seq_len = self.config[key]["seq_len"]
                    n_pred = self.config[key]["n_pred"]

                    options = {}
                    options['output_dir'] = directory
                    options["model_dir"] = model_directory
                    options["model_path"] = options["model_dir"] + "best_bert.pth"
                    options["vocab_path"] = directory + "/vocab.pkl"
                    options["seq_len"] = seq_len
                    options["num_candidates"] = n_pred

                    model = LogBERTAnalyzer(directory=directory, 
                                            model_directory=model_directory, 
                                            seq_len=seq_len)

                    self.log_analyzers[key] = model

                case "DL":
                    model_path = self.config[key]["model"]
                    seq_len = self.config[key]["seq_len"]
                    n_pred = self.config[key]["n_pred"]
                    model = DeepLogAnalyzer(model_path=model_path, 
                                            n_preds=n_pred,
                                            sequence_length=seq_len)                                                
                    self.log_analyzers[key] = model
                case _:
                    raise ValueError(f"Unknown  type: {self.config[key]['type']}")
    
    def get_models(self, key:str):
        if key in self.log_analyzers:
            return self.log_analyzers[key]
        else:
            raise KeyError(f"Model with key {key} not found.")

    def run(self, key:str, val):
        model = self.log_analyzers[key]
        return model.predict_single_sequence(val)
    
