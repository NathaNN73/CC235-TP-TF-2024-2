import cv2
import numpy as np
from skimage.feature import hog

# Conversión a escala de grises
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)    
    gray_blurred = cv2.medianBlur(gray, 5)
    equalized = cv2.equalizeHist(gray_blurred)
    return equalized


# Detección de bordes
def detect_edges(image):

    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
    sobel_combined = cv2.magnitude(sobel_x, sobel_y)
    
    sobel_combined = np.uint8(sobel_combined)
    
    edges = cv2.Canny(sobel_combined, 100, 200)
    
    return edges


# Segmentación por Watershed
def watershed_segmentation(image):
    # Convertir la imagen a escala de grises y aplicar un umbral
    ret, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Transformación de la distancia
    dist_transform = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
    
    # Crear marcadores para el algoritmo Watershed
    ret, markers = cv2.connectedComponents(np.uint8(dist_transform > 0.7 * dist_transform.max()))
    
    # Ajuste para watershed
    img_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    markers = np.int32(markers)
    
    # Aplicar Watershed
    markers = cv2.watershed(img_color, markers)
    

    img_color[markers == -1] = [0, 0, 255]  # Bordes marcados en rojo
    return markers


# Extracción de contornos
def extract_contours(markers):
    markers_bin = np.uint8(markers)
    markers_bin[markers_bin == -1] = 255 
    markers_bin[markers_bin != 255] = 0 
    
    # Encontrar contornos en la imagen binaria
    contours, _ = cv2.findContours(markers_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours

# Extracción de características de forma
def extract_shape_features(contours):
    '''    features = []
    for cnt in contours:
        moments = cv2.moments(cnt)
        hu_moments = cv2.HuMoments(moments).flatten()
        features.append(hu_moments)
    
    return np.array(features)
    '''
    features, _ = hog(contours, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)

    return np.array(features)
