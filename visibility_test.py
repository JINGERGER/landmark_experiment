#!/usr/bin/env python3
"""
简单的物体可见性测试脚本
"""

import cv2
import numpy as np

# 创建测试图像
test_image = np.zeros((600, 800, 3), dtype=np.uint8)

# 测试物体位置和颜色
objects = [
    {"name": "person", "x": 300, "y": 300, "radius": 20, "color": (0, 255, 255)},    # 青色
    {"name": "car", "x": 600, "y": 400, "radius": 40, "color": (0, 0, 255)},        # 红色
    {"name": "bicycle", "x": 400, "y": 140, "radius": 30, "color": (0, 255, 0)},    # 绿色
    {"name": "bottle", "x": 160, "y": 360, "radius": 5, "color": (255, 255, 0)},    # 黄色
    {"name": "chair", "x": 500, "y": 260, "radius": 40, "color": (255, 0, 255)},    # 紫色
]

print("🎨 绘制测试物体:")
for obj in objects:
    # 绘制填充圆
    cv2.circle(test_image, (obj["x"], obj["y"]), obj["radius"], obj["color"], -1)
    # 绘制白色边框
    cv2.circle(test_image, (obj["x"], obj["y"]), obj["radius"] + 3, (255, 255, 255), 3)
    # 绘制黑色中心点
    cv2.circle(test_image, (obj["x"], obj["y"]), 3, (0, 0, 0), -1)
    
    # 添加标签
    cv2.putText(test_image, obj["name"], (obj["x"] - 30, obj["y"] - obj["radius"] - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    print(f"   ✅ {obj['name']}: 位置({obj['x']}, {obj['y']}), 半径={obj['radius']}, 颜色={obj['color']}")

# 保存测试图像
cv2.imwrite("/tmp/visibility_test.jpg", test_image)
print(f"\n📁 测试图像已保存到: /tmp/visibility_test.jpg")

# 检查图像中的非零像素
non_zero_pixels = np.count_nonzero(test_image)
total_pixels = test_image.shape[0] * test_image.shape[1] * test_image.shape[2]
print(f"📊 图像统计: {non_zero_pixels}/{total_pixels} 非零像素 ({non_zero_pixels/total_pixels*100:.2f}%)")
