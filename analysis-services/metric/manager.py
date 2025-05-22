from util import load_config

from metric.dog import MADMetricDog

class MultiDogManager:
    def __init__(self, filepath: str):
        self.config = load_config(filepath)
        self.dogs = {}

        for key in self.config.keys():
            model = self.config[key]["config"]
            match model:
                case "MAD":
                    dog = MADMetricDog(self.config[key]["state"])
                    self.dogs.update({key: dog})
                case _:
                    raise KeyError(f"The model {model} is not known")

    def run(self, key:str, val):
        dog = self.dogs[key]
        return dog.sniff(val)