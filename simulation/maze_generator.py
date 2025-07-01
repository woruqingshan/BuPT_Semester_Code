import numpy as np

def generate_random_maze(width, height, num_segments):
    segments = []
    for _ in range(num_segments):
        if np.random.rand() < 0.5:
            # 垂直线段
            x = np.random.randint(0, width)
            y1 = np.random.randint(0, height)
            y2 = np.random.randint(0, height)
            segments.append((x, y1, x, y2))
        else:
            # 水平线段
            y = np.random.randint(0, height)
            x1 = np.random.randint(0, width)
            x2 = np.random.randint(0, width)
            segments.append((x1, y, x2, y))
    return segments

def segments_to_obstacles(segments, scale_factor=1.0):
    """
    将线段转换为障碍物点数组
    
    Args:
        segments: 线段列表，每个元素为(x1, y1, x2, y2)
        scale_factor: 坐标缩放因子，用于将像素坐标转换为米
        
    Returns:
        numpy.ndarray: 形状为(N, 2)的障碍物坐标数组
    """
    obstacles = []
    for x1, y1, x2, y2 in segments:
        if x1 == x2:  # 垂直线段
            for y in range(min(y1, y2), max(y1, y2) + 1):
                obstacles.append([x1 * scale_factor, y * scale_factor])
        elif y1 == y2:  # 水平线段
            for x in range(min(x1, x2), max(x1, x2) + 1):
                obstacles.append([x * scale_factor, y1 * scale_factor])
    
    if not obstacles:
        # 如果没有障碍物，返回空数组但保持正确的形状
        return np.array([]).reshape(0, 2)
    
    return np.array(obstacles)
