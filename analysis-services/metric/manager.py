from util import load_config

from metric.metric_analyzer import MADMetricAnalyzer

class MultiMAManager:
    def __init__(self, filepath: str):
        self.config = load_config(filepath)
        self.analyzers = {}

        for key in self.config.keys():
            model = self.config[key]["config"]
            match model:
                case "MAD":
                    analyzer = MADMetricAnalyzer(self.config[key]["state"])
                    self.analyzers.update({key: analyzer})
                case _:
                    raise KeyError(f"The model {model} is not known")

    def run(self, key:str, val):
        dog = self.analyzers[key]
        return dog.analyze(val)