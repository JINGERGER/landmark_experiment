#!/usr/bin/env python3
"""
Test script to verify distance filtering parameters
"""

def test_distance_filtering():
    # Test parameters
    min_distance = 0.1
    max_distance = 10.0
    
    # Test cases
    test_depths = [0.22, 0.15, 0.05, 0.62, 1.5, 5.0, 12.0]
    
    print("🔍 Testing distance filtering:")
    print(f"Range: {min_distance}m - {max_distance}m")
    print("-" * 40)
    
    for depth in test_depths:
        if min_distance <= depth <= max_distance:
            status = "✅ PASS"
        else:
            status = "❌ FILTER"
        print(f"{depth:5.2f}m -> {status}")
    
    print("-" * 40)
    print("椅子深度 0.22m 现在应该通过过滤器了！")

if __name__ == '__main__':
    test_distance_filtering()
