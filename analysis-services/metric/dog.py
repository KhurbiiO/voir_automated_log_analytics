import numpy as np
import pickle

from pysad.models import MedianAbsoluteDeviation

class MADMetricDog():
    """
    A metric stream anomaly detection agent based on a Median Absolute Deviation model

    """

    def __init__(self, state_path, threshold=1000):
        self.m = MedianAbsoluteDeviation()
        self.state_path = state_path
        self.threshold = threshold
        self.count = 0

        self.load() # Load model if one exists

    def sniff(self, val):
        """
        Analyze data coming in and fit model for the new value. (Automatic save after dog has sniffed a specified amount of values)

        """
        score = self.m.fit_score_partial(np.array([val], dtype=np.float64))
        self.count += 1
        if self.count == self.threshold:
            self.save(self.state_path)
            self.count = 0

        return score
    
    def parse_data(self, values):
        X = np.array([[x] for x in values], dtype=np.float64) 
        return X
    
    def train(self, val):
        X = self.parse_data(val)
        self.m.fit(X)
    
    def load(self):
        try:
            with open(self.state_path, "rb") as file:
                self.m = pickle.load(file)
        except Exception as e:
            print(e)

    def save(self):
        with open(self.state_path, "wb") as file:
            pickle.dump(self.m, file)

