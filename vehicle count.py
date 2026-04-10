import cv2
from ultralytics import YOLO

model= YOLO('yolov8n.pt')

video_path='dhaka traffic.mp4'
cap=cv2.VideoCapture(video_path)

while cap.isOpened():
    success, frame= cap.read()
    if not success:
        break
    result=model.track(frame, persist=True, verbose=False)
    
    car=bike=bus=truck=0
    if result[0].boxes.cls is not None:
        classes=result[0].boxes.cls.cpu().numpy()
        car=list(classes).count(2)
        bike=list(classes).count(3)
        bus=list(classes).count(5)
        truck=list(classes).count(7)
        
    total_vehicle=car + bike + bus + truck
    print(f"Traffic Report: {car} Cars, {bike} Bikes, {bus} Buses, and {truck} Trucks.")
        
    annotated_frame=result[0].plot()
    cv2.putText(annotated_frame, f"Car: {car}", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),2)
    cv2.imshow("YOLOv8 Counting", annotated_frame)
    
    if cv2.waitKey(1) & 0xFF==ord("s"):
        break
    
cap.release()
cv2.destroyAllWindows()
    