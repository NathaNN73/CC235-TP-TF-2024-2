import numpy as np
from data_processing import DataProcessing

#train data
image_folder = "data/train/images"
augmented_image_folder = "data/train/augmented_images"
label_folder = "data/train/labels"

#valid data
valid_image_folder = "data/valid/images"
valid_label_folder = "data/valid/labels"

#test data
test_image_folder = "data/test/images"
test_label_folder = "data/test/labels"

# Load features and labels

data_processing = DataProcessing()

try:
    print("Loading augmented features and labels")
    augmented_features, augmented_labels = data_processing.get_features_labels("data/train/augmented_images.csv")
except:
    print("Error loading augmented features and labels")
    exit()


print(augmented_features.shape)

np.save("data/train/all_features.npy", augmented_features)
np.save("data/train/all_labels.npy", augmented_labels)

