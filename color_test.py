#!/usr/bin/env python3
"""
测试改进后的颜色方案
"""

import cv2
import numpy as np

def get_class_color(class_id):
    """Get consistent color for each class"""
    # Generate bright, distinguishable colors based on class ID
    color_options = [
        (0, 255, 255),    # 青色 - Cyan
        (255, 0, 255),    # 紫色 - Magenta  
        (0, 255, 0),      # 绿色 - Green
        (255, 255, 0),    # 黄色 - Yellow
        (0, 0, 255),      # 红色 - Red
        (255, 128, 0),    # 橙色 - Orange
        (128, 255, 0),    # 亮绿 - Bright Green
        (255, 0, 128),    # 粉色 - Pink
        (0, 128, 255),    # 天蓝 - Sky Blue
        (128, 0, 255),    # 紫罗兰 - Violet
    ]
    return color_options[class_id % len(color_options)]

# 创建测试图像
test_image = np.zeros((400, 600, 3), dtype=np.uint8)

# 测试颜色
class_names = ["person", "car", "bicycle", "bottle", "chair", "dog", "cat", "bus", "truck", "bird"]

print("🎨 测试改进后的颜色方案:")
for i, name in enumerate(class_names):
    color = get_class_color(i)
    
    # 绘制色块
    x = (i % 5) * 120 + 10
    y = (i // 5) * 180 + 50
    
    cv2.circle(test_image, (x + 50, y + 50), 40, color, -1)
    cv2.circle(test_image, (x + 50, y + 50), 43, (255, 255, 255), 3)
    
    # 标签
    cv2.putText(test_image, name, (x, y + 120),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    print(f"  {i}: {name} -> {color}")

# 保存测试图像
cv2.imwrite("/tmp/color_test.jpg", test_image)
print(f"\n✅ 颜色测试图像已保存到: /tmp/color_test.jpg")
