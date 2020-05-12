import pickle
import urllib.request
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np 
import pandas as pd 
import seaborn as sns
import os 

from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
from keras.models import model_from_json
import cv2

model_masterCategory = pickle.load(open('model_masterCategory_final_hyper.sav','rb'))
model_subCategory = pickle.load(open('model_subCategory_final_hyper.sav','rb'))
model_articleType = pickle.load(open('model_articleType_final_hyper.sav','rb'))
model_gender = pickle.load(open('model_gender_final_hyper.sav','rb'))

pca_masterCategory_n20 = pickle.load(open('pca_masterCategory_n20.sav','rb'))
pca_subCategory_n40 = pickle.load(open('pca_subCategory_n40.sav','rb'))
pca_articleType_n40 = pickle.load(open('pca_articleType_n40.sav','rb'))
pca_gender_n40 = pickle.load(open('pca_gender_n40.sav','rb'))

df = pd.read_csv('final_styles2.csv')
df.drop('Unnamed: 0',axis=1,inplace=True)

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


# functions for image loading and plotting
def plot_figures(figures, nrows = 1, ncols=1,figsize=(8, 8)):  
    fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows,figsize=figsize)
    for ind,title in enumerate(figures):
        axeslist.ravel()[ind].imshow(cv2.cvtColor(figures[title], cv2.COLOR_BGR2RGB))
        axeslist.ravel()[ind].set_title(title)
        axeslist.ravel()[ind].set_axis_off()
    plt.tight_layout() # optional
    


def load_image_url(url, resized_fac = 0.5):
    # download the image, convert it to a NumPy array, and then read
    # it into OpenCV format
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    
    w, h, _ = image.shape
    resized = cv2.resize(image, (int(h*resized_fac), int(w*resized_fac)), interpolation = cv2.INTER_AREA)
    
    # return the image
    return resized

def get_embedding_url(model, url):
    # Reshape
    img = loadImage(url)
    # img to Array
    x   = image.img_to_array(img)
    # Expand Dim (1, w, h)
    x   = np.expand_dims(x, axis=0)
    # Pre process Input
    x   = preprocess_input(x)
    return model.predict(x).reshape(-1)

def loadImage(URL):
    try:
        with urllib.request.urlopen(URL) as url:
            img = image.load_img(BytesIO(url.read()), target_size=(img_width, img_height))
            return image.img_to_array(img)
    except:
        print('HTTP Error 403: Forbidden')
        return False

def predict_by_url(model, url):
    
    # try:
    emb = get_embedding_url(model, url)
    emb = pd.DataFrame(emb).transpose()
    # except:
        # return False

    # Predict Master Category
    feature_pca = pca_masterCategory_n20.transform(emb)
    dfPCA = pd.DataFrame(feature_pca,columns=['PCA'+str(i) for i in range(1,21)])

    pred_master = model_masterCategory.predict(dfPCA)[0]
    proba_master = model_masterCategory.predict_proba(dfPCA).max()*100
    
    # Predict Sub Category
    feature_pca = pca_subCategory_n40.transform(emb)
    dfPCA = pd.DataFrame(feature_pca,columns=['PCA'+str(i) for i in range(1,41)])

    # adding master category prediction result as feature
    dfpred = pd.DataFrame([pred_master],columns=['masterCategory'])
    dfpred = dfpred.append(pd.DataFrame([df['masterCategory'].unique()],index=['masterCategory']).transpose())
    dfpred = pd.DataFrame(pd.get_dummies(dfpred,drop_first=True).iloc[0]).transpose()

    dfPCA = dfPCA.join(dfpred)

    pred_sub = model_subCategory.predict(dfPCA)[0]
    proba_sub = model_subCategory.predict_proba(dfPCA).max()*100
    
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
    proba_art = model_articleType.predict_proba(dfPCA).max()*100
    
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
    proba_gen = model_gender.predict_proba(dfPCA).max()*100
    
    return pred_master, pred_sub, pred_art, pred_gen