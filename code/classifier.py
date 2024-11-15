import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from data_processing import DataProcessing
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib  
from sklearn.model_selection import GridSearchCV

class Classifier:
    def __init__(self, features=None, labels=None):
        self.features = features
        self.labels = labels
        self.classifier = SVC(kernel='linear')
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=50)


        
    def train(self):
        print("Reescalando características...")
        self.features = self.scaler.fit_transform(self.features)
        print("Tamaño de características después del escalado: ", self.features.shape)

        print("Aplicando PCA ...")
        self.features = self.pca.fit_transform(self.features)
        print("Tamaño de características: ", self.features.shape)

        param_grid = [
            {'C': [0.1, 1, 10], 'kernel': ['linear']},
            {'C': [0.1, 1, 10], 'kernel': ['rbf'], 'gamma': ['scale', 'auto', 0.1, 1]}
        ]

        #print("Entrenando modelo...")
        #self.classifier.fit(self.features, self.labels)

        print("Entrenando modelo con GridSearchCV...")
        grid_search = GridSearchCV(self.classifier, param_grid, cv=5, n_jobs=-1, verbose=2)
        grid_search.fit(self.features, self.labels)
        self.classifier = grid_search.best_estimator_

        print("Mejores parámetros encontrados:", grid_search.best_params_)
        print("Mejor puntuación:", grid_search.best_score_)
        
    def evaluate(self, test_image_folder, test_label_folder):
        test_features, test_labels = DataProcessing().get_images_labels_folder(test_image_folder, test_label_folder)
        print("Reescalando características...")
        test_features = self.scaler.transform(test_features)

        print("Aplicando PCA ...")
        test_features = self.pca.transform(test_features)


        predictions = self.classifier.predict(test_features)
        return classification_report(test_labels, predictions)
    
    '''
    def evaluate_validation(self, validation_image_folder, validation_labels_folder):
        validation_features, validation_labels = get_images_labels_folder(validation_image_folder, validation_labels_folder)
        predictions = self.classifier.predict(validation_features)
        return classification_report(validation_labels, predictions)

        
    '''

    def add_features_labels(self, features, labels):
        self.features = np.concatenate((self.features, features), axis=0)
        self.labels = np.concatenate((self.labels, labels), axis=0)

    def save_model(self, path):
        # Guarda el clasificador, el scaler y el PCA en un solo archivo
        joblib.dump((self.classifier, self.scaler, self.pca), path)

    def load_model(self, path):
        # Carga el clasificador, el scaler y el PCA desde el archivo
        self.classifier, self.scaler, self.pca = joblib.load(path)


