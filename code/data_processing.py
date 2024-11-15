import os 
import pandas as pd
import numpy as np
from skimage.feature import hog
from skimage.io import imread
from skimage.color import rgb2gray


class DataProcessing:
    def __init__(self):
        pass

    def extract_features(self, image):
        features, _ = hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
        return features

    def get_features_labels(self, path):
        df = pd.read_csv(path, sep=',')

        features = []
        labels = []

        it = 1
        for image, label in zip(df['image'], df['class']):
            
            if "augmented" in image:
                aux = image.split("_")
                newimage = "_".join(aux[2:])
                image = newimage
                    
            image_path = os.path.join("data/train/images", image)

            if not os.path.exists(image_path):
                print(f"Advertencia: La imagen {image_path} no existe.")
                continue

            image = imread(image_path)
            gray_image = rgb2gray(image)
            image_features = self.extract_features(gray_image)

            if image_features is not None and len(image_features) > 0:
                features.append(image_features)
                labels.append(label)
                print(f"{it}: Extracted features for {image_path}")
                it += 1

            
            #features.append(image)
            #labels.append(label)
            #print(f"{it}: Extracted features for {image_path}")
            
            #if it == 30:
             #   break

        features = np.array(features)
        labels = np.array(labels)

        if features.ndim == 1:
            features = features.reshape(-1, 1)

        return features, labels

    def get_images_labels_folder(self, image_folder, label_folder):
        images = os.listdir(image_folder)
        labels = os.listdir(label_folder)
        features = []
        labels_list  = []

        it = 1
        for image, label in zip(images, labels):
            
            image_path = os.path.join(image_folder, image)
            label_path = os.path.join(label_folder, label)


            if not os.path.exists(image_path):
                print(f"Advertencia: La imagen {image_path} no existe.")
                continue

        

            image = imread(image_path)
            gray_image = rgb2gray(image)
            image_features = self.extract_features(gray_image)

            if image_features is not None and len(image_features) > 0:
                with open(label_path, 'r') as file:
                    aux_file = file.readline()

                    if aux_file == "":
                        print(f"Advertencia: No se pudo leer la etiqueta de {label_path}.")
                        continue
                    else:
                        label_value = int(aux_file.split()[0])
                        labels_list.append(label_value)
                        features.append(image_features)
                        print(f"{it}: Extracted features for {image_path}")
    
            else:
                print(f"Advertencia: No se pudieron extraer características de {image_path}.")
                continue

            
        
            it += 1
            #if it == 20:
             #   break

        features = np.array(features)
        labels_list = np.array(labels_list)

        if features.ndim == 1:
            features = features.reshape(-1, 1)

        return features, labels_list
