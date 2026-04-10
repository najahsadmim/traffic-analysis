# Real-Time Urban Traffic Analysis
A high-performance computer vision pipeline designed to monitor and categorize vehicle density in high-congestion urban environments. This project implements YOLOv8 for real-time object detection and tracking, providing actionable traffic metrics.

**Tech Stack:**
- Language: Python
- Framework: PyTorch
- Model: YOLOv8 (Nano version)
- Libraries: OpenCV (frame processing and UI overlay)

**Features:**
- Real-Time Detection: Processes video streams to identify cars, motorcycles, buses, and trucks.
- Dynamic Tracking: Uses algorithms via Ultralytics to maintain persistent vehicle IDs across frames.
- Optimized Pipeline: Efficiently handles memory transfer between GPU (Tensors) and CPU (NumPy) for low-latency inference.

**Future Improvements:**
- Fine-tune the model on a custom dataset specifically for Rickshaws, CNGs, and Legunas (for South Asian countries like Bangladesh, India, Pakistan)
- Instead of just counting everything in the frame, define a specific coordinate line. A vehicle is only counted when its centroid crosses that line.
