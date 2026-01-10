
# Efficient Attention Mechanisms for Real-Time Projectile Tracking

## Executive Summary

This project compares **Standard Attention** vs **Multi-Query Attention** for real-time object tracking on resource-constrained embedded systems (targeting robotics applications like the T-Sphere disaster response robot).

## Results Summary

### Model Architectures

| Model | Parameters | Embedding Dim | Heads | Depth |
|-------|-----------|--------------|-------|-------|
| Standard Attention | 208,418 | 64 | 4 | 4 |
| Multi-Query Attention | 183,458 | 64 | 4 | 4 |

### Performance Metrics

| Metric | Standard Attention | Multi-Query Attention | Change |
|--------|-------------------|---------------------|--------|
| **Accuracy** (within 2px) | 100.00% | 100.00% | +0.00% |
| **Avg Error** (pixels) | 0.13 | 0.24 | +0.11px |
| **Speed** (FPS) | 1375.15 | 1224.44 | -11.0% |
| **Latency** (ms) | 0.73 | 0.82 | +12.3% |
| **Parameters** | 208,418 | 183,458 | -12.0% |
| **Training Loss** | 0.0112 | 0.0751 | +0.0639 |
| **Validation Loss** | 0.0116 | 0.0380 | +0.0264 |

## Key Findings

### ✅ **Parameter Efficiency** 
Multi-Query Attention reduces parameters by **12.0%** (208,418 → 183,458 parameters)
- Enables deployment on memory-constrained devices
- Lower storage requirements for embedded systems

### ⚠️ **Speed-Accuracy Tradeoff**
- **Speed**: 11.0% slower (1375 FPS → 1224 FPS)
- **Accuracy**: Maintained at 100% (within 2 pixels)
- **Average Error**: Increased from 0.13px to 0.24px (still sub-pixel accurate)

### 🔍 **Training Characteristics**
- Multi-Query has higher training loss (0.0751 vs 0.0112)
- Multi-Query has higher validation loss (0.0380 vs 0.0116)
- Suggests Multi-Query may have slightly reduced capacity but still achieves excellent accuracy

## Analysis

**Standard Attention:**
- ✅ Slightly faster inference (1375 FPS)
- ✅ Lower training/validation loss
- ✅ Marginally better average error (0.13px)
- ❌ 12% more parameters

**Multi-Query Attention:**
- ✅ 12% fewer parameters (better for embedded systems)
- ✅ Maintains 100% accuracy on 2-pixel threshold
- ❌ 11% slower (but still very fast at 1224 FPS)
- ❌ Slightly higher error (0.24px vs 0.13px)

## Conclusion

**Both models achieve excellent performance**, with different strengths:

**For T-Sphere Robotics Application:**

The choice depends on system constraints:

**Choose Standard Attention if:**
- Maximum speed is critical (1375 FPS vs 1224 FPS)
- Memory is not constrained
- Highest precision needed (0.13px vs 0.24px error)

**Choose Multi-Query Attention if:**
- Memory/storage is limited (12% smaller model)
- 1224 FPS is sufficient for real-time tracking
- 0.24px error is acceptable (still sub-pixel accurate)

**Recommendation**: For the T-Sphere embedded system (Jetson Nano), **Multi-Query Attention** is preferable due to its smaller memory footprint, even though it's slightly slower. Both models achieve 100% accuracy and >1000 FPS, making either suitable for real-time tracking.

## Real-World Context

- Both models achieve **>1000 FPS** - far exceeding real-time requirements (30-60 FPS)
- Both maintain **100% accuracy** within 2-pixel threshold
- The 11% speed difference (150 FPS) is negligible for this application
- The 12% parameter reduction could be critical for embedded deployment

**Verdict**: Multi-Query Attention offers a favorable tradeoff for embedded robotics.
