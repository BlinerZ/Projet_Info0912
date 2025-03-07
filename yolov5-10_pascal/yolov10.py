from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

model = YOLO("yolov10n.pt")

image_path = "yolov5-10_pascal/image_personne.jpg"

results = model(image_path)

for result in results:
    img = result.plot()

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 6))
    plt.imshow(img_rgb)
    plt.axis("off")
    plt.show()
