import os
import joblib
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

class ClassicalBaselines:
    def __init__(self):
        self.models = {
            'svm_rbf': SVC(kernel='rbf', C=1.0, probability=True, random_state=42),
            'svm_linear': SVC(kernel='linear', C=1.0, probability=True, random_state=42),
            'logistic_regression': LogisticRegression(C=1.0, max_iter=1000, random_state=42),
            'mlp': MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=1000, random_state=42)
        }
        self.is_fitted = False

    def fit(self, X, y):
        """
        Train all classical models on the exact same feature matrix X and labels y.
        """
        for name, model in self.models.items():
            model.fit(X, y)
        self.is_fitted = True
        return self

    def predict_all(self, x):
        """
        Predict with all classical baselines.
        x: single sample (n_features,) or batch (N, n_features)
        Returns dictionary with predicted classes and probabilities.
        """
        if not self.is_fitted:
            raise RuntimeError("Classical baselines are not fitted yet.")

        is_single = (x.ndim == 1)
        if is_single:
            x = x.reshape(1, -1)

        results = {}
        for name, model in self.models.items():
            probs = model.predict_proba(x)[:, 1] # Probability of Pneumonia
            preds = (probs >= 0.5).astype(int)
            if is_single:
                results[name] = {
                    'prediction': int(preds[0]),
                    'class_name': "Pneumonia" if preds[0] == 1 else "Normal",
                    'probability': float(probs[0])
                }
            else:
                results[name] = {
                    'predictions': preds.tolist(),
                    'probabilities': probs.tolist()
                }
        return results

    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            'models': self.models,
            'is_fitted': self.is_fitted
        }, filepath)

    @classmethod
    def load(cls, filepath):
        data = joblib.load(filepath)
        baselines = cls()
        baselines.models = data['models']
        baselines.is_fitted = data['is_fitted']
        return baselines


DEFAULT_FINDINGS = [
    "Infiltration",
    "Effusion",
    "Atelectasis",
    "Pneumothorax",
    "Pneumonia",
    "Cardiomegaly",
    "Consolidation",
    "No Finding"
]

class MultiDiseaseClassicalBaselines:
    """
    Classical Multi-Output Baselines for multi-disease thoracic pathology finding detection.
    Evaluated on the exact same orthogonal PCA features.
    """
    def __init__(self, finding_names=None):
        self.finding_names = finding_names or list(DEFAULT_FINDINGS)
        self.lr_estimators = {}
        self.mlp = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=1000, random_state=42)
        self.is_fitted = False

    def fit(self, X, Y):
        """
        X: (N, n_features)
        Y: (N, n_findings) binary matrix
        """
        for i, name in enumerate(self.finding_names):
            y_col = Y[:, i]
            unique_classes = np.unique(y_col)
            if len(unique_classes) > 1:
                lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
                lr.fit(X, y_col)
                self.lr_estimators[name] = lr
            else:
                # If only one class present in training split, store constant
                self.lr_estimators[name] = float(unique_classes[0])

        self.mlp.fit(X, Y)
        self.is_fitted = True
        return self

    def predict_all(self, x, threshold=0.50):
        if not self.is_fitted:
            raise RuntimeError("MultiDiseaseClassicalBaselines are not fitted yet.")

        is_single = (x.ndim == 1)
        if is_single:
            x = x.reshape(1, -1)

        results = {}

        # 1. Multi-Output Logistic Regression
        lr_probs = []
        for i, name in enumerate(self.finding_names):
            est = self.lr_estimators.get(name)
            if isinstance(est, (float, int)):
                p = np.full(x.shape[0], float(est))
            elif est is not None:
                if len(est.classes_) == 1:
                    p = np.full(x.shape[0], 1.0 if est.classes_[0] == 1 else 0.0)
                else:
                    p = est.predict_proba(x)[:, 1]
            else:
                p = np.full(x.shape[0], 0.1)
            lr_probs.append(p)
        lr_probs = np.column_stack(lr_probs)

        # 2. Multi-label MLP
        mlp = self.mlp
        mlp_probs = mlp.predict_proba(x)

        if is_single:
            results['logistic_regression'] = {
                'probabilities': {
                    self.finding_names[i]: round(float(lr_probs[0, i]), 4)
                    for i in range(len(self.finding_names))
                },
                'detected_findings': [
                    self.finding_names[i]
                    for i in range(len(self.finding_names))
                    if lr_probs[0, i] >= threshold
                ]
            }
            results['mlp'] = {
                'probabilities': {
                    self.finding_names[i]: round(float(mlp_probs[0, i]), 4)
                    for i in range(len(self.finding_names))
                },
                'detected_findings': [
                    self.finding_names[i]
                    for i in range(len(self.finding_names))
                    if mlp_probs[0, i] >= threshold
                ]
            }
        else:
            results['logistic_regression'] = {
                'probabilities': lr_probs.tolist(),
                'predictions': (lr_probs >= threshold).astype(int).tolist()
            }
            results['mlp'] = {
                'probabilities': mlp_probs.tolist(),
                'predictions': (mlp_probs >= threshold).astype(int).tolist()
            }

        return results

    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            'lr_estimators': self.lr_estimators,
            'mlp': self.mlp,
            'finding_names': self.finding_names,
            'is_fitted': self.is_fitted
        }, filepath)

    @classmethod
    def load(cls, filepath):
        data = joblib.load(filepath)
        baselines = cls(finding_names=data.get('finding_names'))
        baselines.lr_estimators = data.get('lr_estimators', {})
        baselines.mlp = data.get('mlp')
        baselines.is_fitted = data.get('is_fitted', False)
        return baselines

