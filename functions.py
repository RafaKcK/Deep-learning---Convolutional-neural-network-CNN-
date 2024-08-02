from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Cargamos el modelo previamente entrenado
model = load_model('proyecto_final_cnn.h5')

def predict_image(image_file):

    # Realizamos la predicción sobre una imagen y devolvemos el resultado.
    img = image.load_img(image_file, target_size=(150, 150))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.0  # Normalizamos los valores de píxeles entre (0 ~ 1)

    # Predecimos
    prediction = model.predict(img)
    class_names = ['dew', 'fogsmog', 'frost', 'hail', 'lightning', 'rain', 'sandstorm', 'snow']
    predicted_class = class_names[np.argmax(prediction)]

    return predicted_class