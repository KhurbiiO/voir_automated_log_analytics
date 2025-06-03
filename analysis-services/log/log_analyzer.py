import pandas as pd
import numpy as np
import torch

from bert_pytorch import Predictor

from deeplog import DeepLog 
from deeplog.preprocessor import Preprocessor

class LogBERTAnalyzer:
    def __init__(self, directory: str, model_directory: str, seq_len: int = 100):
        options = {}
        options['output_dir'] = directory
        options["model_dir"] = model_directory
        options["model_path"] = options["model_dir"] + "best_bert.pth"
        options["vocab_path"] = directory + "/vocab.pkl"
        options["seq_len"] = seq_len

        self.options = options
        self.model = Predictor(options)

    def predict_single_sequence(self, logseq) -> dict:
        self.model.predict_single_sequence(logseq)

class DeepLogAnalyzer:
    def __init__(self, model_path: str, n_preds: int = 5, sequence_length: int = 100, device: str = "cpu"):
        self.model = DeepLog.load(model_path, device=device)
        self.sequence_length = sequence_length
        self.n_preds = n_preds
        self.preprocessor = Preprocessor(length=self.sequence_length, timeout=float("inf"))

    def process_data(self, df:pd.DataFrame) -> list:
        df["event"] = df["Template"]
        df["timestamp"] = pd.to_datetime(df["@timestamp"], format='ISO8601')
        df['timestamp'] = df["timestamp"].values.astype(np.int64) // 10 ** 9
        df["machine"] = "NULL"
        return df[["timestamp", "event", "machine"]]

    def calculate_anomaly_score(self, preds, ground_truth) -> float:
        correct= 0
        for i in range(len(preds)):
            if ground_truth[i] in preds[i]:
                correct += 1
        return correct / len(preds)

    def change_sequence_length(self, new_length: int):
        self.sequence_length = new_length
        self.preprocessor = Preprocessor(length=self.sequence_length, timeout=float("inf"))

    def predict_single_sequence(self, df) -> dict:
        df = self.process_data(df)
        X, y, _, _ = self.preprocessor.sequence(df)
        y_pred, confidence = self.model.predict(
            X = X,
            k = self.n_preds,
        )

        # Convert tensors to numpy arrays
        y_pred = y_pred.cpu().numpy()
        y = y.cpu().numpy()
        confidence = confidence.cpu().numpy()

        return self.calculate_anomaly_score(y_pred, y), y_pred.tolist(), confidence.tolist()