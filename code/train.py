import numpy as np
from classifier import Classifier

def train_classifier():
    features = np.load("source/train/all_features.npy")
    labels = np.load("source/train/all_labels.npy")
   

    # Crear y entrenar el clasificador
    classifier = Classifier(features, labels)
    classifier.train()

    # Evaluar el clasificador

    #print("Resultados en el conjunto de prueba:")
    #print(classifier.evaluate("source/test/images", "source/test/labels"))

    #Evaluar el clasificador en el conjunto de validación
    print("Resultados en el conjunto de validación:")
    print(classifier.evaluate("source/valid/images", "source/valid/labels"))

    #model = Classifier()
    #model.load_model('source/model/classifier_50_grid.h5')
    #print(model.evaluate("source/valid/images", "source/valid/labels"))

    # Guardar el clasificador
    classifier.save_model("source/model/classifier_50_grid.h5")
    print("Modelo guardado exitosamente.")

train_classifier()
