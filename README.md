# Practicum2
MSDS 692
## Objective:
The purpose of this study is to identify if we can accurately predict is a breast tumor is malignant or benign with the data collected in this data set.  In the future we might be able to uses models similar to this to help predict the outcome of medical conditions based off of previous information that we have seen in the past.

## Data:
This data was pulled from https://www.kaggle.com/uciml/breast-cancer-wisconsin-data.  The contains 569 lines of data.  This data is broken down into 32 attributes with one of them being the diagnosis, which we are attempting to predict.  Out of these 569 lines of data, 357 of the diagnosis are benign and 212 are malignant.  This data set is rather small, but we should be able do create a model that is able to accurately predict is the tumor is cancerous or benign.  
The type of data science task that I used was Classification using supervised learning to determine the diagnosis of the tumor. I will be using the 32 attributes to attempt to identify what attributes are most helpful in determining outcome of the tumor, is it cancerous or not.

## Attribute Information:
1) ID number
2) Diagnosis (M = malignant, B = benign)
3-32)
Ten real-valued features are computed for each cell nucleus:
a) radius (mean of distances from center to points on the perimeter)
b) texture (standard deviation of gray-scale values)
c) perimeter
d) area
e) smoothness (local variation in radius lengths)
f) compactness (perimeter^2 / area - 1.0)
g) concavity (severity of concave portions of the contour)
h) concave points (number of concave portions of the contour)
i) symmetry
j) fractal dimension ("coastline approximation" - 1)
The mean, standard error and "worst" or largest (mean of the three
largest values) of these features were computed for each image,
resulting in 30 features. For instance, field 3 is Mean Radius, field
13 is Radius SE, field 23 is Worst Radius.
All feature values are recoded with four significant digits.

## Data Collection and Preparation:
I pulled this data from https://www.kaggle.com/uciml/breast-cancer-wisconsin-data and pulled this into a data frame to analyze it in python.  The amount of data cleaning looks to be minimal but there are a few things that we need to do. 
