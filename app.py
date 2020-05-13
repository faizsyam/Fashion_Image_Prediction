from flask import Flask, render_template, request, redirect, Response, url_for
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
# from flask_dropzone import Dropzone
from prediction import predict_by_url, init
from werkzeug import secure_filename
import os
import io
import random
import cv2

UPLOAD_FOLDER = '/Purwadhika/FINAL PROJECT/dashboard/static'

# Translate Flask to Python object
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

global model
model = init()

@app.route('/', methods=['GET','POST'])
def index_prediction():
    if request.method == 'POST':
        if request.files:
            image = request.files["imageBrowse"]
            if image.filename=="":
                data = request.form
                data = data.to_dict()
                hasil = predict_by_url(model,data['imageURL'])
                if not hasil:
                    return render_template('prediction.html',imageInfo='Image not found or filetype not supported.')
                else:            
                    probas = [str(round(hasil[i],2)) for i in range(4,8)]
                    print(probas)
                    return render_template('result.html',hasil_prediction=hasil,url_prediction=data['imageURL'],prediction_probas=probas)
            else:
                filename = secure_filename(image.filename)
                image.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
                hasil = predict_by_url(model,filename)
                if not hasil:
                    return render_template('prediction.html',imageInfo='Image not found or filetype not supported.')
                else:            
                    probas = [str(round(hasil[i],2)) for i in range(4,8)]
                    print(probas)  
                    return render_template('result.html',hasil_prediction=hasil,url_prediction=url_for('static', filename=filename),prediction_probas=probas)
        
    return render_template('prediction.html')

@app.route('/prediction')
def prediction():
    return render_template('prediction.html')

@app.route('/about')
def about():
    return render_template('about.html')
    
@app.route('/dataset')
def dataset():
    return render_template('dataset.html')

if __name__ == '__main__':
    app.run(debug=True,port=1122)