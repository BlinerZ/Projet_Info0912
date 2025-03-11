# PROJET_INFO0912

- ARNOUDTS Kevin
- BRZYCHCY Loïc
- DARVILLE Killian

Utilisation de YoloV5 et YoloV10 pour la détection d'objets sur une base de données de maladies de plantes.

## Structure du projet

- `runs/` : Dossier généré par Ultralytics contenant les métriques d'entraînement et d'évaluation
  - `runs/yoloV10_basePlant/data` : Résultats pour YOLO V10 sur plantdoc basique
  - `runs/yoloV10_basePlant/aug` : Résultats pour YOLO V10 sur plantdoc augmenté
  - `runs/yoloV5_basePlant/data` : Résultats pour YOLO V5 sur plantdoc basique
  - `runs/yoloV5_basePlant/aug` : Résultats pour YOLO V5 sur plantdoc augmenté
- `yolov5-10_pascal/` : Code source Ultralytics basé sur PASCAL VOC
- `custom_data_yolov10.yaml` : Configuration pour plantdoc basique
- `custom_augdata_yolov10.yaml` : Configuration pour plantdoc augmenté
- `labels_generator.py` : Script pour générer les boîtes englobantes au bon format pour Ultralytics
- Scripts d'exécution des modèles yolo :
  - `yoloV5_plant.py` : YOLO V5 sur plantdoc basique
  - `yoloV5_aug_plant.py` : YOLO V5 sur plantdoc augmenté
  - `yoloV10_plant.py` : YOLO V10 sur plantdoc basique
  - `yoloV10_aug_plant.py` : YOLO V10 sur plantdoc augmenté


## Résultats

Les résultats de l'entraînement et de l'évaluation sont stockés dans le dossier `runs/`, pour YoloV5 et YoloV10 sur la base plantdoc basique et augmenté.