import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

model = tf.keras.models.load_model("brain_tumor_model.h5")

class_names = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]

img_path = input("Enter image path: ")

img = image.load_img(img_path, target_size=(128, 128))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

prediction = model.predict(img_array)

predicted_class = class_names[np.argmax(prediction)]

print("\nPrediction:", predicted_class)