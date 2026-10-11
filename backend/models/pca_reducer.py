import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

class PCAReducer:
    def __init__(self, n_components=4):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components, random_state=42)
        self.scaler = MinMaxScaler(feature_range=(0.0, np.pi))
        self.is_fitted = False

    def fit(self, features):
        """
        Fits PCA on high-dimensional feature matrix (N, 512) and fits MinMax scaler.
        """
        pca_feats = self.pca.fit_transform(features)
        self.scaler.fit(pca_feats)
        self.is_fitted = True
        return self

    def transform(self, features):
        """
        Transforms features (N, 512) or (512,) into quantum angle-scaled features (N, n_components).
        Returns:
            scaled_features: array bounded in [0, pi] suitable for AngleEmbedding (Ry rotation)
            raw_pca: unscaled PCA coordinates
            explained_variance: list of explained variance per component
        """
        if not self.is_fitted:
            raise RuntimeError("PCA model is not fitted yet.")
        
        is_single = (features.ndim == 1)
        if is_single:
            features = features.reshape(1, -1)

        raw_pca = self.pca.transform(features)
        scaled = self.scaler.transform(raw_pca)

        exp_var = self.pca.explained_variance_ratio_.tolist()
        
        if is_single:
            return scaled[0], raw_pca[0], exp_var
        return scaled, raw_pca, exp_var

    def get_explained_variance_ratio(self):
        if not self.is_fitted:
            return []
        return self.pca.explained_variance_ratio_.tolist()

    def get_total_explained_variance(self):
        if not self.is_fitted:
            return 0.0
        return float(np.sum(self.pca.explained_variance_ratio_))

    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            'n_components': self.n_components,
            'pca': self.pca,
            'scaler': self.scaler,
            'is_fitted': self.is_fitted
        }, filepath)

    @classmethod
    def load(cls, filepath):
        data = joblib.load(filepath)
        reducer = cls(n_components=data['n_components'])
        reducer.pca = data['pca']
        reducer.scaler = data['scaler']
        reducer.is_fitted = data['is_fitted']
        return reducer
