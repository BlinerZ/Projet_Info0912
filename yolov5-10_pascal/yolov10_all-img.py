from ultralytics import YOLO
import cv2
import os

model = YOLO("yolov10n.pt")

input_folder = "yolov5-10_pascal/img/input"
output_folder = "yolov5-10_pascal/img/output/v10"

os.makedirs(output_folder, exist_ok=True)

image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

for image_file in image_files:
    image_path = os.path.join(input_folder, image_file)
    
    results = model(image_path)
    
    for result in results:
        img = result.plot()
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        output_path = os.path.join(output_folder, f"{image_file}")
        cv2.imwrite(output_path, img)
