from ultralytics import YOLO
model = YOLO('yolov5m.pt')
# Entraîner sur votre dataset
model.train(
    data="custom_data_yolov10.yaml",  # Chemin vers le fichier YAML
    epochs=100,  # Nombre d'époques
    batch=16,  # Taille du lot
    imgsz=640,  # Taille des images
    workers=4,  # Nombre de processus pour le traitement
)