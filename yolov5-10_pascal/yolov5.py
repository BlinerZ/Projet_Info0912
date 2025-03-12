from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

model = YOLO("yolov5n.pt") #Chargement modèle préentrainer COCO (Pascal et COCO étant similaire, pas besoin de réentrainement pour avoir de bon résultats)

image_path = "yolov5-10_pascal/image-02.png" #Image sur laquelle inférerer

results = model(image_path) #Réalise l'inférence

for result in results: #Affiche les résultats dans une fenetre dédié (matplotlib)
    img = result.plot()

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 6))
    plt.imshow(img_rgb)
    plt.axis("off")
    plt.show()
