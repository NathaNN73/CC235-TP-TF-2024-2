import cv2
import os
import pandas as pd
from image_variation import create_image_variations

def readfile(path):
    files = os.listdir(path) 
    return files

def writefile(path, content):
    cv2.imwrite(path, content)


df_images = pd.DataFrame(columns=["image", "class"])
df_auemented_images = pd.DataFrame(columns=["image", "class"])


images = readfile("data/train/images")
labels = readfile("data/train/labels")


image_rows = []
augmented_image_rows = []

it = 0

for image in images:
    img = cv2.imread("data/train/images/" + image)
    label_path = "data/train/labels/" + image.replace(".jpg", ".txt")

    try:
        with open(label_path, 'r') as file:
            class_id = int(file.readline().split()[0])
    except:
        class_id = -1
        pass
    

    if class_id != -1 and it % 4 == 0:
        image_rows.append({"image": image, "class": class_id})
        for i in range(5):
            name_file = "augmented_#" + str(i) + "_" + image
            augmented_img = create_image_variations(img)
            writefile("data/train/augmented_images/" + name_file, augmented_img)
            augmented_image_rows.append({"image": name_file, "class": class_id})
        
        print(f"{it} Image {image} processed")
    
    it += 1


#  DataFrames para images y augmented_images
df_images = pd.DataFrame(image_rows)
df_auemented_images = pd.DataFrame(augmented_image_rows)

# export 
df_images.to_csv("data/train/images.csv", index=False)
df_auemented_images.to_csv("data/train/augmented_images.csv", index=False)