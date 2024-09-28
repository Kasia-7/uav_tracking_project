import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import cm
from scipy.spatial.distance import pdist, squareform

# 设置中文字体，解决乱码问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为 SimHei 显示中文
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

def visualize_3d_trajectory_with_time(trajectory, cluster_data, radar_position=(0, 0, 0), step=5):
    """
    可视化无人机集群的3D轨迹及雷达位置，并使用颜色渐变来表示时间的流逝。
    计算集群内无人机之间的距离变化，显示不同时间段的集群规模变化。
    
    Parameters:
    - trajectory: List of tuples (time, (x, y, z))，群中心轨迹。
    - cluster_data: pandas DataFrame, 包含每个时间点的无人机位置 (斜距(m), 方位角（°）, 俯仰角（°）)。
    - radar_position: 雷达的位置，默认在 (0, 0, 0)。
    - step: 间隔的步长，只显示部分时间点上的轨迹。
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 提取群中心轨迹，并应用颜色渐变
    times = [t[0] for t in trajectory][::step]
    x_centroid = [t[1][0] for t in trajectory][::step]
    y_centroid = [t[1][1] for t in trajectory][::step]
    z_centroid = [t[1][2] for t in trajectory][::step]

    # 使用颜色渐变表示时间
    norm = plt.Normalize(min(times), max(times))
    cmap = cm.get_cmap('viridis')

    for i in range(len(times) - 1):
        ax.plot(x_centroid[i:i + 2], y_centroid[i:i + 2], z_centroid[i:i + 2],
                color=cmap(norm(times[i])), linewidth=2)

    # 可视化雷达的位置
    ax.scatter(*radar_position, color='b', s=100, label='雷达位置')

    # 处理无人机集群的轨迹：根据step取部分时间点进行显示，颜色渐变
    for time_step in sorted(cluster_data['时间(s)'].unique())[::step]:
        current_data = cluster_data[cluster_data['时间(s)'] == time_step]
        x_vals = current_data['斜距(m)'].values
        y_vals = current_data['方位角（°）'].values
        z_vals = current_data['俯仰角（°）'].values
        
        # 使用渐变色显示无人机位置
        ax.scatter(x_vals, y_vals, z_vals, color=cmap(norm(time_step)), s=20, alpha=0.6)

        # 计算无人机之间的距离并显示
        if len(current_data) > 1:  # 确保有多个无人机可以计算距离
            distances = pdist(current_data[['斜距(m)', '方位角（°）', '俯仰角（°）']].values)
            mean_distance = np.mean(distances)
            min_distance = np.min(distances)
            max_distance = np.max(distances)
            print(f"时间 {time_step}s: 平均距离={mean_distance:.2f}, 最小距离={min_distance:.2f}, 最大距离={max_distance:.2f}")
        else:
            print(f"时间 {time_step}s: 无法计算距离，只有一个无人机点。")

    # 设置坐标轴标签和图例
    ax.set_xlabel('斜距(m)')
    ax.set_ylabel('方位角（°）')
    ax.set_zlabel('俯仰角（°）')
    plt.title('无人机集群3D轨迹及雷达位置')

    # 添加颜色条表示时间
    mappable = cm.ScalarMappable(cmap=cmap, norm=norm)
    mappable.set_array(times)
    plt.colorbar(mappable, label='时间 (s)')
    
    plt.savefig("figures/results_2.png")
    plt.show()


def analyze_cluster_size(cluster_data, step=5):
    """
    通过不同时间段估算集群规模，验证集群规模估计的准确性。
    
    Parameters:
    - cluster_data: pandas DataFrame, 包含每个时间点的无人机位置 (斜距(m), 方位角（°）, 俯仰角（°）)。
    - step: 每隔多少个时间点进行规模估算。
    """
    from sklearn.cluster import DBSCAN
    
    cluster_sizes = []
    time_steps = sorted(cluster_data['时间(s)'].unique())
    
    for time_step in time_steps[::step]:
        current_data = cluster_data[cluster_data['时间(s)'] == time_step]
        features = current_data[['斜距(m)', '方位角（°）']].values
        db = DBSCAN(eps=5, min_samples=3).fit(features)  # 调整 eps 和 min_samples
        labels = db.labels_
        num_clusters = len(set(labels)) - (1 if -1 in labels else 0)  # 忽略噪声点
        
        cluster_sizes.append((time_step, num_clusters))
        # print(f"时间 {time_step}s: 估计的群规模={num_clusters}")
    
    # 绘制不同时间段的集群规模变化
    times, sizes = zip(*cluster_sizes)
    plt.plot(times, sizes, marker='o', linestyle='-')
    plt.xlabel('时间 (s)')
    plt.ylabel('估计的群规模')
    plt.title('集群规模随时间的变化')
    plt.show()
