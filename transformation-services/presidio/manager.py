from util import load_config

from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
from presidio_analyzer.nlp_engine import NlpEngineProvider

class MultiPresidioManager:
    def __init__(self, filepath: str):
        self.config = load_config(filepath)
        self.engines = {}

        for key in self.config.keys():
            provider = NlpEngineProvider(conf_file=self.config[key]["path"])
            nlp_engine = provider.create_engine()

            analyzer = AnalyzerEngine(
                nlp_engine=nlp_engine,
                supported_languages=[key]
            )

            self.engines.update({key: analyzer})
            
    def get_engine(self, key):
        return self.engines[key]

    def run(self, lang:str, msg: str):
        engine = self.engines[lang]
        results = engine.analyze(text=msg,
                           language=lang)
        return [msg[pred.start:pred.end] for pred in results]
        

