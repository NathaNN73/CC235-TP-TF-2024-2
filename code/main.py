# video_processing.py
import cv2
from image_feature_extraction import preprocess_image, \
    detect_edges, watershed_segmentation, extract_shape_features, \
    extract_contours


from classifier import Classifier
from skimage.feature import hog 

def process_video(video_path, model):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 30.0, (640, 480))
    
    print("Processing video...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Pipeline completo
        preprocessed = preprocess_image(frame)
        #mask = segment_by_color(frame)
        edges = detect_edges(preprocessed)
        markers = watershed_segmentation(edges)
        
        # Extraer y clasificar características
        contours = extract_contours(markers)
        #contours, _ = cv2.findContours(markers, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        #features = extract_shape_features(contours)

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        features, _ = hog(gray_frame, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
        print(features.shape)

        features = features.reshape(1, -1)
        features = model.scaler.transform(features)
        features = model.pca.transform(features)
        predictions = model.classifier.predict(features)
        
        # Dibujar las detecciones en el frame
        for cnt, label in zip(contours, predictions):
            if label == 1:  # Si es una señal de tránsito
                cv2.drawContours(frame, [cnt], 0, (0, 255, 0), 2)
        
        out.write(frame)
        cv2.imshow('Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Cargar el modelo del clasificador
model = Classifier()
model.load_model('data/model/classifier_1000.h5')

process_video('data/video/video.mp4', model)



