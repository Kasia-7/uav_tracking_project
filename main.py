# main.py
from src.data_loader import DataLoader
from src.preprocessing import MissingValueFillerPreprocessing
from src.feature_extraction import FeatureExtractor
from src.tracking import CentroidTracking
from src.clustering import ClusterEstimator
from src.utils import visualize_3d_trajectory_with_time, analyze_cluster_size

def main():
    # 1. 加载数据
    loader = DataLoader(file_path='data/点迹数据3-公开提供.xlsx', file_type='excel')
    data = loader.load_data()

    # 2. 数据预处理
    preprocessor = MissingValueFillerPreprocessing()
    preprocessed_data = preprocessor.preprocess(data)

    # 分析集群规模随时间的变化
    analyze_cluster_size(preprocessed_data, step=5)

    # 3. 特征提取
    # feature_extractor = FeatureExtractor(preprocessed_data)
    # features = feature_extractor.extract_basic_features()

    # 4. 群中心跟踪
    tracker = CentroidTracking(preprocessed_data)
    trajectory = tracker.track_centroid_trajectory()
    # 可视化群中心轨迹和无人机轨迹
    visualize_3d_trajectory_with_time(trajectory, preprocessed_data, radar_position=(0, 0, 0), step=50)
    # 打印质心轨迹
    # print("群中心轨迹：")
    # for time_step, centroid in trajectory:
    #     print(f"时间: {time_step} - 群中心位置: {centroid}")

    # 5. 群规模估计
    cluster_estimator = ClusterEstimator(eps=5, min_samples=3)
    cluster_labels, num_clusters = cluster_estimator.estimate(preprocessed_data)

    print(f"估计的群规模: {num_clusters}")

if __name__ == '__main__':
    main()
