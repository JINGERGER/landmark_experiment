#!/usr/bin/env python3
"""
测试BEV物体大小可视化的演示脚本
"""

import cv2
import numpy as np

def draw_bev_with_sizes():
    """创建一个BEV图像演示不同大小的物体"""
    
    # BEV参数
    bev_width = 800
    bev_height = 600
    bev_x_range = 4.0  # 4米前方
    bev_y_range = 3.0  # 左右各1.5米
    
    # 创建黑色背景
    bev_image = np.zeros((bev_height, bev_width, 3), dtype=np.uint8)
    
    # 绘制网格
    grid_spacing = 0.25  # 25cm
    
    # 垂直线（距离标记）
    for x_m in np.arange(0, bev_x_range + grid_spacing, grid_spacing):
        x_pixel = int((x_m / bev_x_range) * bev_width)
        if x_pixel < bev_width:
            if x_m % 1.0 == 0:  # 主网格线每1米
                cv2.line(bev_image, (x_pixel, 0), (x_pixel, bev_height), (100, 100, 100), 2)
                if x_m > 0:
                    cv2.putText(bev_image, f"{x_m:.0f}m", (x_pixel + 5, 20),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
            else:  # 次网格线
                cv2.line(bev_image, (x_pixel, 0), (x_pixel, bev_height), (60, 60, 60), 1)
    
    # 水平线（左右标记）
    for y_m in np.arange(-bev_y_range/2, bev_y_range/2 + grid_spacing, grid_spacing):
        y_pixel = int(bev_height - ((y_m + bev_y_range/2) / bev_y_range) * bev_height)
        if 0 <= y_pixel < bev_height:
            if abs(y_m) % 1.0 == 0:  # 主网格线
                cv2.line(bev_image, (0, y_pixel), (bev_width, y_pixel), (100, 100, 100), 2)
                if abs(y_m) > 0.1:
                    cv2.putText(bev_image, f"{y_m:+.0f}m", (5, y_pixel - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
            else:  # 次网格线
                cv2.line(bev_image, (0, y_pixel), (bev_width, y_pixel), (60, 60, 60), 1)
    
    # 中心线（摄像头位置）
    center_y = bev_height // 2
    cv2.line(bev_image, (0, center_y), (bev_width, center_y), (0, 255, 0), 2)
    cv2.putText(bev_image, "Camera", (10, center_y - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # 添加尺寸参考图例
    legend_x = 10
    legend_y = 50
    cv2.putText(bev_image, "Size Reference:", (legend_x, legend_y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    # 绘制不同尺寸的参考圆
    ref_sizes = [0.1, 0.3, 0.5, 1.0]  # 米
    for i, size_m in enumerate(ref_sizes):
        ref_radius_px = max(2, int((size_m / bev_x_range) * bev_width * 0.5))
        ref_x = legend_x + 20 + i * 60
        ref_y = legend_y + 25
        
        cv2.circle(bev_image, (ref_x, ref_y), ref_radius_px, (150, 150, 150), 1)
        cv2.putText(bev_image, f"{size_m}m", (ref_x - 10, ref_y + ref_radius_px + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)
    
    # 模拟一些不同大小的检测物体 - 放在网格绘制之后
    detections = [
        {"class": "person", "x": 1.5, "y": 0.0, "size": 0.4, "color": (0, 255, 255)},    # 人 - 青色
        {"class": "car", "x": 3.0, "y": -0.5, "size": 1.2, "color": (0, 0, 255)},       # 车 - 红色
        {"class": "bicycle", "x": 2.0, "y": 0.8, "size": 0.6, "color": (0, 255, 0)},    # 自行车 - 绿色
        {"class": "bottle", "x": 0.8, "y": -0.3, "size": 0.1, "color": (255, 255, 0)},  # 瓶子 - 黄色
        {"class": "chair", "x": 2.5, "y": 0.2, "size": 0.8, "color": (255, 0, 255)},    # 椅子 - 紫色
    ]
    
    for det in detections:
        # 计算BEV像素坐标
        bev_x = det["x"]  # 前方距离
        bev_y = det["y"]  # 左右偏移
        
        bev_pixel_x = int((bev_x / bev_x_range) * bev_width)
        bev_pixel_y = int(bev_height - ((bev_y + bev_y_range/2) / bev_y_range) * bev_height)
        
        # 计算物体在BEV中的大小
        equivalent_radius_m = det["size"] / 2
        bev_radius_px = max(3, int((equivalent_radius_m / bev_x_range) * bev_width * 0.5))
        bev_radius_px = min(bev_radius_px, 40)  # 限制最大尺寸
        
        # 调试信息
        print(f"🔍 {det['class']}: 世界坐标({bev_x}, {bev_y}) -> 像素坐标({bev_pixel_x}, {bev_pixel_y}), 半径={bev_radius_px}px")
        
        # 检查物体是否在图像范围内
        if not (0 <= bev_pixel_x < bev_width and 0 <= bev_pixel_y < bev_height):
            print(f"⚠️  {det['class']} 超出图像范围!")
            continue
        
        # 绘制物体
        color = det["color"]
        cv2.circle(bev_image, (bev_pixel_x, bev_pixel_y), bev_radius_px, color, -1)
        cv2.circle(bev_image, (bev_pixel_x, bev_pixel_y), bev_radius_px + 3, (255, 255, 255), 3)  # 更粗的白色边框
        cv2.circle(bev_image, (bev_pixel_x, bev_pixel_y), 3, (0, 0, 0), -1)  # 更大的黑色中心点
        
        # 添加标签
        label = det["class"]
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
        label_x = max(0, min(bev_pixel_x - label_size[0]//2, bev_width - label_size[0]))
        label_y = max(15, bev_pixel_y - bev_radius_px - 20)
        
        cv2.rectangle(bev_image, 
                     (label_x - 2, label_y - 12),
                     (label_x + label_size[0] + 2, label_y + 2),
                     (0, 0, 0), -1)
        cv2.putText(bev_image, label, (label_x, label_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # 添加距离和尺寸信息
        distance = np.sqrt(bev_x**2 + bev_y**2)
        info_text = f"{distance:.1f}m ⌀{det['size']:.2f}m"
        
        info_size = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)[0]
        info_x = max(0, min(bev_pixel_x - info_size[0]//2, bev_width - info_size[0]))
        info_y = min(bev_height - 5, bev_pixel_y + bev_radius_px + 15)
        
        cv2.rectangle(bev_image,
                     (info_x - 1, info_y - 10),
                     (info_x + info_size[0] + 1, info_y + 2),
                     (0, 0, 0), -1)
        cv2.putText(bev_image, info_text, (info_x, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
    
    return bev_image

if __name__ == "__main__":
    print("🎯 生成BEV物体大小可视化演示...")
    
    # 创建BEV演示图像
    bev_demo = draw_bev_with_sizes()
    
    # 保存图像
    output_path = "/tmp/bev_size_demo.jpg"
    cv2.imwrite(output_path, bev_demo)
    print(f"✅ BEV演示图像已保存到: {output_path}")
    
    # 显示图像尺寸和内容说明
    print(f"📏 图像尺寸: {bev_demo.shape[1]}x{bev_demo.shape[0]}")
    print("📋 演示内容:")
    print("   - 网格线：每25cm一条次线，每1m一条主线")
    print("   - 尺寸图例：显示0.1m, 0.3m, 0.5m, 1.0m的参考圆")
    print("   - 模拟物体：人(黄)、车(蓝)、自行车(绿)、瓶子(青)、椅子(紫)")
    print("   - 信息显示：物体类别、距离、直径")
    print("   - 圆圈大小：与物体实际尺寸成比例")
