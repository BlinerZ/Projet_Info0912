from ultralytics import YOLO
model = YOLO('yolov5m.pt')  #Chargement poids yolov5 préentrainer COCO (git ultralytics)

model.train(
    data="custom_data_yolov10.yaml",  # Chemin vers le fichier YAML, permet de fournir les classes et les dossiers contenant les données
    epochs=100,  # Défini grâce au surentrainement et early stopping du premier modèle tel que présenter dans la slide
    batch=16,  
    imgsz=640,  # Taille des images
)