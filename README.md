# Fashion Product Images Classification
#### By: Faizuddarain Syam

Online marketplaces are getting huge. Customers and sellers are on its raise. Various new online retails are introduced every year all around the world. With this huge market, it is important to keep everything in control. We must ensure that every retailer are doing their part well, by not abusing the system for their own benefit or even misinforming the customers by the product they are selling.

Categorizing products is one of the simpliest yet critical mistake. By having a system that can alert or give suggestion to the seller, not only it can prevent mistakes, but also makes the experience for sellers to upload their products better.

------------------------------------

- In this project we will be predicting a fashion product image's category and gender. The product' s category will be divided into three, Main Category, Sub Category, and Article Type. The learning algorithm that we will be using is only limited to Random Forest and XGBoost.
- Dataset source: https://www.kaggle.com/paramaggarwal/fashion-product-images-dataset

## Part 1: Preprocessing

In this first part, we will be focusing on what should we be doing to the data to make things easier and more optimum before taking it into the processing stage. Since this is a huge dataset, our main objective is to how we reduce the amount of data without taking out any important information.

### Removing categories with low amount of data
Since the smallest combination of data we are aiming in this project is articleType and Gender. We will be removing the combinations of both of them which its amount of data are less than a minimum value (min_amount). This value is believed to be the minimum amount of data per class a model needs to be trained and tested. 

### Undersampling
The data look better, but still has unbalanced amount of data on certain categories. So the first step we will be looking into articleType and gender combination with amounts that are above our determined maximum amount.

### Embedding Images

Image embedding is when we convert high-dimensional images into a relatively low-dimensional space, making machine learning models easier to process. In this part, we will convert our image data into an array of numbers. After image is embedded, it is then processed by a pre-trained deep learning model, ResNet50.

Basically, ResNet50 is a convolutional neural network (CNN) model which is 50 layers deep. The model is pre-trained, which means it has already be trained by thousands of image to be classified. With this model, we will be translating our image data into a numeric array. This array will then be used to train our classification model.

## Part 2: Training

After our data are ready to go. Now we need to design a learning model which can bring out the best from our data. We will be analyzing two different algorithms, Random Forest and XGBoost Classifier. And also we will be doing additional process on our data before trained. Maybe reducing it's features or oversample the low amount classes? We'll see.

### Calculating Accuracy Scores for various n components values on PCA
Our plan here is to find the best accuracy score for each category with various n-components of PCA. Not only that, we will be also try to oversample our data using SMOTE. Let's which model turns out to be the best for each category.

### Adding larger categories as training feature for smaller categories
Because there will be an additional categorical data, we will be using get dummies to convert it into numerical

Results:
- Master Category : XGBoost, with SMOTE, PCA n=20
- Sub Category : Random Forest, with SMOTE, PCA n=40
- Article Type : Random Forest, with SMOTE, PCA n=40
- Gender : XGBoost, without SMOTE, PCA n=40

### Hyperparameter on selected models
From the models above we will further optimize them using Hyperparameter

#### Final results:

**Master Category : XGBoost, with SMOTE, PCA n=20**

learning_rate': 0.3, 'max_depth': 4, 'min_child_weight': 1
Accuracy score: 0.99

**Sub Category : Random Forest, with SMOTE, PCA n=40**

'max_depth': 60, 'min_samples_leaf': 1, 'min_samples_split': 2
Accuracy score: 0.95

**Article Type : Random Forest, with SMOTE, PCA n=40**

'max_depth': 60, 'min_samples_leaf': 1, 'min_samples_split': 2
Accuracy score: 0.87

**Gender : XGBoost, without SMOTE, PCA n=40**

'learning_rate': 0.3, 'max_depth': 8, 'min_child_weight': 1
Accuracy score: 0.91

**Average accuracy score: 0.93**

# Part 3: Application

