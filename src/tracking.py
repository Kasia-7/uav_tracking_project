# tracking.py
import numpy as np

class KalmanFilter:
    def __init__(self, init_position, init_velocity=0):
        """初始化卡尔曼滤波器，init_position 和 init_velocity 应该是数值或 numpy 数组"""
        self.position = np.array(init_position, dtype=float)  # 将初始位置转换为 numpy 数组
        self.velocity = np.array(init_velocity, dtype=float)

    def predict(self, delta_time):
        """预测下一时刻的位置"""
        self.position += self.velocity * delta_time
        return self.position

    def update(self, measured_position):
        """使用测量值更新位置"""
        measured_position = np.array(measured_position, dtype=float)  # 将测量位置转换为 numpy 数组
        self.position = 0.5 * self.position + 0.5 * measured_position  # 简单加权更新
        return self.position

# class CentroidTracking:
#     def __init__(self, points):
#         """初始化质心跟踪器，points 为包含每个点位置信息的字典列表"""
#         self.points = points

#     def calculate_centroid(self):
#         """计算无人机群的中心位置"""
#         x_mean = np.mean(self.points['斜距(m)'])
#         y_mean = np.mean(self.points['方位角（°）'])
#         z_mean = np.mean(self.points['俯仰角（°）'])
#         return x_mean, y_mean, z_mean

#     def track_with_kalman(self, init_position, delta_time):
#         """结合卡尔曼滤波跟踪群中心位置"""
#         kf = KalmanFilter(init_position=init_position)
#         centroid = self.calculate_centroid()
#         return kf.update(centroid)

class CentroidTracking:
    def __init__(self, data, time_column='时间(s)'):
        """
        初始化质心跟踪器。
        data 应为 pandas DataFrame，time_column 为时间戳的列名。
        """
        self.data = data
        self.time_column = time_column
        self.time_steps = sorted(self.data[time_column].unique())  # 获取所有时间步长

    def calculate_centroid_at_time(self, time_step):
        """计算给定时间点的质心"""
        current_data = self.data[self.data[self.time_column] == time_step]
        x_mean = np.mean(current_data['斜距(m)'])
        y_mean = np.mean(current_data['方位角（°）'])
        z_mean = np.mean(current_data['俯仰角（°）'])
        return x_mean, y_mean, z_mean

    def track_centroid_trajectory(self):
        """计算所有时间点上的质心，形成质心轨迹"""
        trajectory = []
        for time_step in self.time_steps:
            centroid = self.calculate_centroid_at_time(time_step)
            trajectory.append((time_step, centroid))  # 将时间戳和质心一起记录
        return trajectory
