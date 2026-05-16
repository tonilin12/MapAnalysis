# Map Sheet Corner & OCR Detection

Computer vision project for automatic inner-corner detection and sheet identifier recognition from scanned historical topographic map sheets.

---

# Overview

This project processes scanned 1950s map sheets and automatically extracts:

- upper-left corner
- upper-right corner
- lower-right corner
- lower-left corner
- sheet identifier

The system is designed for large-scale batch processing and evaluation on historical scanned maps.

---

# Features

- automatic corner detection
- automatic sheet ID recognition
- batch image processing
- debug visualization outputs
- result export
- robust handling of noisy scans

---

# Technologies

- Python
- OpenCV
- NumPy
- PaddleOCR

---

# Performance

| Metric | Result |
|---|---|
| Corner detection | ~5 px error for >95% of sheets |
| OCR accuracy | ~95% exact identifier accuracy |

---
