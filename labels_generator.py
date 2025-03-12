import os
import cv2

def generate_bounding_boxes(mask_path: str) -> None:
    """
    Crée les BBOX à partir da le méthode findContours
    
    Args:
        mask_path (str): Le lien vers le mask de l'image
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    # Détècte les contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bounding_boxes = []
    for contour in contours:
        # Récupère les coordonnées de chaque contour
        x, y, w, h = cv2.boundingRect(contour)
        bounding_boxes.append([float(x), float(y), float(w), float(h)])

    return bounding_boxes

# Dossiers source et destination
base_path = "augdata"
mask_folders = {
    "train": "augdata/train/masks",
    "test": "augdata/test/masks"
}

for dataset_type, mask_folder in mask_folders.items():
    label_output_folder = os.path.join(base_path, dataset_type, "labels")

    # Créer les dossiers s'ils n'existent pas
    os.makedirs(label_output_folder, exist_ok=True)

    mask_files = os.listdir(mask_folder)

    for mask_file in mask_files:
        mask_path = os.path.join(mask_folder, mask_file)
        bounding_boxes = generate_bounding_boxes(mask_path)

        # Obtenir les dimensions de l'image associée
        img = cv2.imread(mask_path)
        if img is None:
            print(f"Image non trouvée ou invalide : {mask_path}")
            continue
        height, width, _ = img.shape

        # Nom du fichier .txt correspondant
        label_file = os.path.join(label_output_folder, mask_file.replace(".png", ".txt"))

        with open(label_file, "w") as f:
            for box in bounding_boxes:
                # Calcul des coordonnées normalisées pour YOLO
                x, y, w, h = box
                center_x = (x + w / 2) / width
                center_y = (y + h / 2) / height
                norm_width = w / width
                norm_height = h / height

                # Écrire dans le fichier avec class_id (0 ici, à modifier si nécessaire)
                f.write(f"0 {center_x:.6f} {center_y:.6f} {norm_width:.6f} {norm_height:.6f}\n")

print("Conversion terminée.")