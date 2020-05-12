from flask import Flask, render_template, request, redirect
from flask_dropzone import Dropzone
from prediction import predict_by_url, init
import os

# Translate Flask to Python object
app = Flask(__name__)
dropzone = Dropzone(app)

global model

model = init()

@app.route('/', methods=['GET','POST'])
def index_prediction():
    if request.method == 'POST':
        # if request.files:
        #     image = request.files["image"]
        #     print(image)
        #     return render_template('result.html',hasil_prediction=image)
        # else:
        data = request.form
        data = data.to_dict()
        # hasil = data['imageURL']
        hasil = predict_by_url(model,data['imageURL'])
        # if not hasil:
        #     return render_template('about.html')
        # else:
        return render_template('result.html',hasil_prediction=hasil)
        # return render_template('result.html')
    return render_template('prediction.html')

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True,port=1010)