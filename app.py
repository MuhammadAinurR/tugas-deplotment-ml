from flask import Flask, render_template, request, url_for

from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input, decode_predictions, VGG16

app = Flask(__name__)
model = VGG16()

@app.route('/', methods=['GET'])
def hello_word():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    image_path = "./static/image.jpg" 
    imagefile.save(image_path)

    image = load_img(image_path, target_size=[224, 224])
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    imae = preprocess_input(image)
    yhat = model.predict(image)
    label = decode_predictions(yhat)
    label = label[0][0]
    classification = '%s (%.2f%%)' % (label[1], label[2]*100)

    return render_template('index.html', prediction = classification)

if __name__ == '__main__':
    app.run(port=5000, debug=True)