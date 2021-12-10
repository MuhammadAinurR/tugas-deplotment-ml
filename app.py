from flask import Flask, render_template, request, url_for

from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model
import pickle
import numpy as np
from tensorflow.keras.applications.vgg16 import preprocess_input
import os

app = Flask(__name__)
global model
model = load_model(os.path.join("model.h5"))
with open("labels.txt", "rb") as fp:
    labels = pickle.load(fp)
labels[1]

import pickle
with open("labels.txt", "rb") as fp:
    labels = pickle.load(fp)
labels[1]

@app.route('/', methods=['GET'])
def hello_word():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    image_path = "./static/image.jpg" 
    imagefile.save(image_path)

    image = load_img(image_path, target_size=[100, 100])
    image = np.asarray(image)
    image = preprocess_input(image)
    image = image.reshape(-1, 100, 100, 3)
    p = model.predict(image)
    p = np.argmax(p, axis=1)
    classification = '%s ' % labels[p[0]]

    return render_template('index.html', prediction = classification)

if __name__ == '__main__':
    app.run(debug=True)