# clustering.py
from sklearn.cluster import DBSCAN
import numpy as np

class ClusterEstimator:
    def __init__(self, eps=5, min_samples=3):
        """初始化聚类模型，默认使用 DBSCAN"""
        self.model = DBSCAN(eps=eps, min_samples=min_samples)

    def estimate(self, data):
        """估算集群规模，data 为 pandas DataFrame"""
        # 正确地使用 DataFrame 列访问方式
        features = data[['斜距(m)', '方位角（°）']].to_numpy()  # 将 DataFrame 转换为 NumPy 数组
        cluster_labels = self.model.fit_predict(features)
        num_clusters = len(set(cluster_labels) - {-1})  # -1 表示噪声
        return cluster_labels, num_clusters
