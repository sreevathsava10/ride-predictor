from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import classification_report

class Classifier(ABC):
    @abstractmethod
    def train(self, *params) -> None:
        pass

    @abstractmethod
    def evaluate(self, *params) -> Dict[str, float]:
        pass

    @abstractmethod
    def predict(self, *params) -> np.ndarray:
        pass


class SklearnClassifier(Classifier):
    def __init__(
        self, estimator: BaseEstimator, features: List[str], target: str,
    ):
        self.clf = estimator
        self.features = features
        self.target = target

    def train(self, df_train: pd.DataFrame):
        self.clf.fit(df_train[self.features].values, df_train[self.target].values)

    def evaluate(self, df_test: pd.DataFrame):
        y_true = df_test[self.target].values

        # Get the predicted probability for the positive class (1)
        pos_class_index = self.clf.classes_.tolist().index(1)
        print("Classifier classes:", self.clf.classes_)
        print("y_true distribution:", pd.Series(y_true).value_counts())
        print("pos_class_index:", pos_class_index)
        y_pred = self.clf.predict_proba(df_test[self.features].values)[:, pos_class_index]
        y_pred_proba = self.clf.predict_proba(df_test[self.features].values)
        
        y_pred_labels = self.clf.predict(df_test[self.features].values)
        class_report = classification_report(y_true, y_pred_labels, output_dict=True)

        # Return the metrics as a dictionary
        return {"classification_report": class_report}

    def predict(self, df: pd.DataFrame):
        return self.clf.predict_proba(df[self.features].values)[:, 1]
