import os

import flask_restful
import numpy as np
import cv2

from PIL import Image
from keras.models import load_model
from flask import Flask, request, render_template
from flask_restful import reqparse, Api, Resource
from werkzeug.utils import secure_filename

app = Flask(__name__)
api = Api(app)

# model = load_model('Brian_Tumor10Epochs.h5')
# print('Model loaded. Check http://127.0.0.1:5000/')


def get_className(classNo):
    if classNo == 0:
        return "No Brain Tumor"
    elif classNo == 1:
        return "Yes Brain Tumor"


def getResult(img):
    image = cv2.imread(img)
    image = Image.fromarray(image, 'RGB')
    image = image.resize((64, 64))
    image = np.array(image)
    input_img = np.expand_dims(image, axis=0)
    predictions = (model.predict(input_img) > 0.5).astype("int32")
    return predictions


# @app.route('/', methods=['GET'])
# def index():
#     return render_template('index.html')


# @app.route('/predict', methods=['GET', 'POST'])
# def upload():
#     if request.method == 'POST':
#         f = request.files['file']
#
#         basepath = os.path.dirname(__file__)
#         file_path = os.path.join(
#             basepath, 'uploads', secure_filename(f.filename))
#         f.save(file_path)
#         value = getResult(file_path)
#         result = get_className(value)
#         return result
#     return None


class Predict(Resource):
    def post(self):
        f = request.files['file']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        value = getResult(file_path)
        result = get_className(value)
        return result


api.add_resource(Predict, '/predict')

if __name__ == '__main__':
    app.run(debug=True)
