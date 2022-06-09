import cv2
import numpy as np

from keras.models import load_model
from PIL import Image

model = load_model('Brian_Tumor10EpochsCategorical.h5')
image = cv2.imread('/home/aka/un/project/brian/pred/pred1.jpg')

img = Image.fromarray(image)

img = img.resize((64, 64))

img = np.array(img)

input_img = np.expand_dims(img, axis=0)

# if Brian_Tumor10EpochsCategorical
predict_x = model.predict(input_img)
classes_x = np.argmax(predict_x, axis=1)

# if Brian_Tumor10Epochs
# predictions = (model.predict(input_img) > 0.5).astype("int32")

print(classes_x)
# print(predictions)
