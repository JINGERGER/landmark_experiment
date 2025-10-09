#!/usr/bin/env python3
"""
Test script to demonstrate the difference between average and minimum depth
"""

import numpy as np

def test_depth_calculation():
    # Simulate depth values for an object (in mm)
    # This represents depth values from a mask region
    depth_values = np.array([220, 225, 230, 235, 240, 245, 250, 280, 300])
    
    print("🔍 Depth calculation comparison:")
    print(f"Depth values (mm): {depth_values}")
    print("-" * 40)
    
    # Calculate average (median) depth - old method
    avg_depth_mm = np.median(depth_values)
    avg_depth_m = avg_depth_mm / 1000.0
    
    # Calculate minimum depth - new method
    min_depth_mm = np.min(depth_values)
    min_depth_m = min_depth_mm / 1000.0
    
    print(f"Average depth: {avg_depth_mm:.0f}mm ({avg_depth_m:.3f}m)")
    print(f"Minimum depth: {min_depth_mm:.0f}mm ({min_depth_m:.3f}m)")
    print(f"Difference:    {avg_depth_mm - min_depth_mm:.0f}mm ({avg_depth_m - min_depth_m:.3f}m)")
    
    print("-" * 40)
    print("✅ Now using minimum depth for closer object detection!")
    print("🎯 Objects will appear at their closest point to camera")

if __name__ == '__main__':
    test_depth_calculation()
