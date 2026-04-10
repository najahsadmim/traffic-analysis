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

**Challenges & Future Improvements:**
- Challenge: The model frequently identifies CNGs/Auto-rickshaws as cars. This occurs because the standard YOLOv8 model was trained on Western datasets that lack three-wheeled motorized vehicles.
  Improvement: Fine-tune the model on a custom dataset specifically for Rickshaws, CNGs, and Legunas (for South Asian countries like Bangladesh, India, Pakistan)
- Challenge: Temporal Consistency, as in ensuring the tracker maintains a unique ID despite high-speed motion and occlusion (objects blocking each other), and eliminates double counting (however, major challenge still remains- if a larger vehicle continues to block the smaller vehicle till the reference line, the vision engine will never recognise the smaller vehicle!)
- Improvement: Instead of just counting everything in the frame, define a specific coordinate line. A vehicle is only counted when its centroid crosses that line.
