from flask import Flask, render_template, request, redirect, Response, url_for
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
# from flask_dropzone import Dropzone
from prediction import predict_by_url, init, show_proba_plot, df
from plot import get_bar_plots, scatter_cat, dfPCA
from werkzeug import secure_filename
import pandas as pd
import os
import io
import random
import cv2
from PIL import Image

UPLOAD_FOLDER = '/Purwadhika/FINAL PROJECT/dashboard/static'

# Translate Flask to Python object
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

global model
model = init()

def get_proba_plots(proba):
    return [show_proba_plot('masterCategory',proba),
    show_proba_plot('subCategory',proba),
    show_proba_plot('articleType',proba),
    show_proba_plot('gender',proba)]

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
                    probas = [str(round(hasil[i].max()*100,2)) for i in range(4,8)]
                    return render_template('result.html',hasil_prediction=hasil,
                    url_prediction=data['imageURL'],prediction_probas=probas,
                    proba_plots=get_proba_plots(hasil))
            else:
                filename = secure_filename(image.filename)
                file_path = os.path.join(app.config["UPLOAD_FOLDER"],filename)
                image_pil = Image.open(image)
                image_pil.save(file_path)
                hasil = predict_by_url(model,image)
                if not hasil:
                    return render_template('prediction.html',imageInfo='Image not found or filetype not supported.')
                else:            
                    probas = [str(round(hasil[i].max()*100,2)) for i in range(4,8)]
                    return render_template('result.html',hasil_prediction=hasil,
                    url_prediction=url_for('static', filename=filename),prediction_probas=probas,
                    proba_plots=get_proba_plots(hasil))
        
    return render_template('prediction.html')

@app.route('/prediction')
def prediction():
    return render_template('prediction.html')

@app.route('/about')
def about():
    return render_template('about.html')
    

@app.route('/dataset')
def dataset():
    newdf = df.sample(200,random_state=101)

    scat_master=scatter_cat('masterCategory',newdf)
    scat_sub=scatter_cat('subCategory',newdf)
    scat_art=scatter_cat('articleType',newdf)
    scat_gender=scatter_cat('gender',newdf)

    plots=get_bar_plots()
    return render_template('dataset.html', df=newdf,
    plot_master=plots[0], plot_sub=plots[1],
    plot_art=plots[2], plot_gen=plots[3],
    scat_master=scat_master,scat_sub=scat_sub,
    scat_art=scat_art,scat_gender=scat_gender)

if __name__ == '__main__':
    app.run(debug=True,port=1123)