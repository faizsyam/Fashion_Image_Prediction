import pickle
import urllib.request
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import numpy as np 
import pandas as pd 
import seaborn as sns
import os 
from io import BytesIO
import base64

from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
from keras.models import model_from_json
import cv2

# prevent gui crash
import matplotlib
matplotlib.use('Agg')

model_masterCategory = pickle.load(open('model_masterCategory_final_hyper.sav','rb'))
model_subCategory = pickle.load(open('model_subCategory_final_hyper.sav','rb'))
model_articleType = pickle.load(open('model_articleType_final_hyper.sav','rb'))
model_gender = pickle.load(open('model_gender_final_hyper.sav','rb'))

pca_masterCategory_n20 = pickle.load(open('pca_masterCategory_n20.sav','rb'))
pca_subCategory_n40 = pickle.load(open('pca_subCategory_n40.sav','rb'))
pca_articleType_n40 = pickle.load(open('pca_articleType_n40.sav','rb'))
pca_gender_n40 = pickle.load(open('pca_gender_n40.sav','rb'))

# Input Shape
img_width, img_height = 224, 224

def init():
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    loaded_model._make_predict_function()
    print("Loaded model from disk")

    return loaded_model

df = pd.read_csv('final_styles2.csv')
df.drop('Unnamed: 0',axis=1,inplace=True)

def get_embedding_url(model, url):
    img = loadImage(url)
    x   = image.img_to_array(img)
    x   = np.expand_dims(x, axis=0)
    x   = preprocess_input(x)
    
    return model.predict(x).reshape(-1)

def loadImage(URL):
    try:
        img = image.load_img(URL, target_size=(img_width, img_height))
    except:
        with urllib.request.urlopen(URL) as url:
            img = image.load_img(BytesIO(url.read()), target_size=(img_width, img_height))
    return img

def predict_by_url(model, url):
    
    try:
            emb = get_embedding_url(model, url)
            emb = pd.DataFrame(emb).transpose()
    except:
        return False

    # Predict Master Category
    feature_pca = pca_masterCategory_n20.transform(emb)
    dfPCA = pd.DataFrame(feature_pca,columns=['PCA'+str(i) for i in range(1,21)])

    pred_master = model_masterCategory.predict(dfPCA)[0]
    proba_master = model_masterCategory.predict_proba(dfPCA)
    
    # Predict Sub Category
    feature_pca = pca_subCategory_n40.transform(emb)
    dfPCA = pd.DataFrame(feature_pca,columns=['PCA'+str(i) for i in range(1,41)])

    # adding master category prediction result as feature
    dfpred = pd.DataFrame([pred_master],columns=['masterCategory'])
    dfpred = dfpred.append(pd.DataFrame([df['masterCategory'].unique()],index=['masterCategory']).transpose())
    dfpred = pd.DataFrame(pd.get_dummies(dfpred,drop_first=True).iloc[0]).transpose()

    dfPCA = dfPCA.join(dfpred)

    pred_sub = model_subCategory.predict(dfPCA)[0]
    proba_sub = model_subCategory.predict_proba(dfPCA)
    
    # Predict Article Type
    feature_pca = pca_articleType_n40.transform(emb)
    dfPCA = pd.DataFrame(feature_pca,columns=['PCA'+str(i) for i in range(1,41)])

    # adding master category prediction result as feature
    dfpred = pd.DataFrame([pred_master],columns=['masterCategory'])
    dfpred = dfpred.append(pd.DataFrame([df['masterCategory'].unique()],index=['masterCategory']).transpose())
    dfpred = pd.DataFrame(pd.get_dummies(dfpred,drop_first=True).iloc[0]).transpose()

    dfPCA = dfPCA.join(dfpred)

    # adding sub category prediction result as feature
    dfpred = pd.DataFrame([pred_sub],columns=['subCategory'])
    dfpred = dfpred.append(pd.DataFrame([df['subCategory'].unique()],index=['subCategory']).transpose())
    dfpred = pd.DataFrame(pd.get_dummies(dfpred,drop_first=True).iloc[0]).transpose()

    dfPCA = dfPCA.join(dfpred)

    pred_art = model_articleType.predict(dfPCA)[0]
    proba_art = model_articleType.predict_proba(dfPCA)
    
    # Predict Gender
    feature_pca = pca_gender_n40.transform(emb)
    dfPCA = pd.DataFrame(feature_pca,columns=['PCA'+str(i) for i in range(1,41)])

    # adding master category prediction result as feature
    dfpred = pd.DataFrame([pred_master],columns=['masterCategory'])
    dfpred = dfpred.append(pd.DataFrame([df['masterCategory'].unique()],index=['masterCategory']).transpose())
    dfpred = pd.DataFrame(pd.get_dummies(dfpred,drop_first=True).iloc[0]).transpose()

    dfPCA = dfPCA.join(dfpred)

    # adding sub category prediction result as feature
    dfpred = pd.DataFrame([pred_sub],columns=['subCategory'])
    dfpred = dfpred.append(pd.DataFrame([df['subCategory'].unique()],index=['subCategory']).transpose())
    dfpred = pd.DataFrame(pd.get_dummies(dfpred,drop_first=True).iloc[0]).transpose()

    dfPCA = dfPCA.join(dfpred)

    # adding article type prediction result as feature
    dfpred = pd.DataFrame([pred_art],columns=['articleType'])
    dfpred = dfpred.append(pd.DataFrame([df['articleType'].unique()],index=['articleType']).transpose())
    dfpred = pd.DataFrame(pd.get_dummies(dfpred,drop_first=True).iloc[0]).transpose()

    dfPCA = dfPCA.join(dfpred)

    pred_gen = model_gender.predict(dfPCA)[0]
    proba_gen = model_gender.predict_proba(dfPCA)
    
    return pred_master, pred_sub, pred_art, pred_gen, proba_master, proba_sub, proba_art, proba_gen

def show_proba_plot(col,pred):
    if col=='masterCategory':
        idx=0
        title='Master Category'
    elif col=='subCategory':
        idx=1
        title='Sub Category'
    elif col=='articleType':
        idx=2
        title='Article Type'
    elif col=='gender':
        idx=3
        title='Gender'
    else:
        return False
    dfg = pd.DataFrame(pred[4+idx]*100,index=['proba'],columns=sorted(df[col].unique()))
    dfg = dfg.transpose().reset_index().sort_values('proba',ascending=False).head(10)
    dfg.rename(columns={'index':col},inplace=True)
    
    img = BytesIO()

    plt.figure(figsize=(10,5))
    plt.title(title)

    sns.set(style="whitegrid")
    g = sns.barplot(x='proba',y=col,data=dfg)
    for p in g.patches:
        plt.text(2+p.get_width(), p.get_y()+0.55*p.get_height(),str(round(p.get_width(),3))+'%',ha='left', va='center')
    g.set_ylabel('')    
    g.set_xlabel('Probability')

    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)

    plot_url = base64.encodebytes(img.getvalue()).decode()

    return plot_url
