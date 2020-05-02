# Practicum2
MSDS 696

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

The mean, standard error and "worst" or largest (mean of the three largest values) of these features were computed for each image, resulting in 30 features. For instance, field 3 is Mean Radius, field 13 is Radius SE, field 23 is Worst Radius.

All feature values are recoded with four significant digits.

## Data Collection and Preparation:
I pulled this data from https://www.kaggle.com/uciml/breast-cancer-wisconsin-data and pulled this into a data frame to analyze it in python.  The amount of data cleaning looks to be minimal but there are a few things that we need to do. 

![Image description](https://github.com/JohnRegis/practicum2/blob/master/1.PNG)

![Image description](https://github.com/JohnRegis/practicum2/blob/master/2.PNG)

The first thing I needed to do was drop a column that had no data.

![Image description](https://github.com/JohnRegis/practicum2/blob/master/3.PNG)

I then needed to change the diagnosis from letters to numbers so I could use this information in my model.

![Image description](https://github.com/JohnRegis/practicum2/blob/master/4.PNG)

The libraries that I need to do this project are the following:
import matplotlib.pyplot as plt, from sklearn.neighbors import KNeighborsRegressor, from sklearn.neighbors import KNeighborsClassifier, from sklearn.model_selection import train_test_split, from sklearn.metrics import confusion_matrix, from sklearn.metrics import accuracy_score, from sklearn.model_selection import train_test_split, from sklearn.ensemble import RandomForestClassifier, import seaborn as sns, from sklearn.metrics import cohen_kappa_score, from sklearn.metrics import confusion_matrix, from sklearn.metrics import accuracy_score, from sklearn.metrics import classification_report, from sklearn import tree, from IPython.display import Image, from sklearn.tree import DecisionTreeClassifier, import io, import pydotplus, import collections, from graphviz import Digraph, from sklearn import metrics, from pprint import pprint, import numpy as np, from sklearn.model_selection import GridSearchCV


First, I wanted to visualize the number of diagnosis of if the tumor was cancer or benign.
![Image description](https://github.com/JohnRegis/practicum2/blob/master/5.PNG)

The first data analysis that I did was to create a heat map: _ = sns.heatmap(df.corr()) here we can see the correlation between the attributes in this dataset.
![Image description](https://github.com/JohnRegis/practicum2/blob/master/6.PNG)

Next, I looked at a pearsoncorr to see the correlations of attributes to the diagnosis.

![Image description](https://github.com/JohnRegis/practicum2/blob/master/7-1.PNG)

I then did the Spearmancorr to see if the correlations of attributes to the diagnosis were different than the pearsoncorr.

![Image description](https://github.com/JohnRegis/practicum2/blob/master/8-1.PNG)

I then created a histogram for all of the features.

![Image description](https://github.com/JohnRegis/Practicum2/blob/master/9.png)

![Image description](https://github.com/JohnRegis/Practicum2/blob/master/10.png)

Then I wanted to see a scatter plot of all of the attributes on how they relate to the diagnosis.  This way we can see how attributes relation to other attributes relate to the diagnosis of the tumor.

![Image description](https://github.com/JohnRegis/practicum2/blob/master/11.PNG)

![Image description](https://github.com/JohnRegis/Practicum2/blob/master/12.png)

The next thing that I did was to look at a scatter plot of the most important feature and one of the least important features to see if I could visualize why the heat map has decided to pick these features. 

From the heatmaps above it appears that the true most important feature with an average correlation score of 0.7896165 is perimeter_worst. 
The here is a scatter plot of the most important feature on predicting diagnosis:

![Image description](https://github.com/JohnRegis/practicum2/blob/master/13.PNG)

From this simple plot we can see that the larger the perimeter at its worst the more likely it is to be cancer.  You can see that there is quite a bit of overlap from around 75 to around 125, in these cases there is a lot of overlap between cancer or not cancer.  For this reason, we cannot just go off of the perimeter at its worst to determine if the tumor is cancer

The here is a scatter plot of the one of the least important features on predicting diagnosis:

![Image description](https://github.com/JohnRegis/practicum2/blob/master/14.PNG)

From this simple plot we can see that not all of the attributes are useful in identifying if the tumor is cancer or now going off of the fractal dimension worst we can see that the majority of this data overlaps and does not help us identify if the tumor is cancer or not.


## Modeling:

The first type of modeling that I did was a simple KNeighborsRegressor I then did a differs to see how far off the guesses (model’s predictions) were from the actuals. Overall, it had the accuracy of 0.7894736842105263.  I did the Cohen Kappa Score for this model to see if this model is due to pure chance, the Cohen Kappa Score is 0.5571382324376821.  With a Cohens Kappa Score between 0.41 – 0.60 we can see that this model has a moderate agreement. This shows that this model is moderately due to the logic of the model and not based off of chance

![Image description](https://github.com/JohnRegis/practicum2/blob/master/15.PNG)

![Image description](https://github.com/JohnRegis/practicum2/blob/master/16.PNG)

![Image description](https://github.com/JohnRegis/practicum2/blob/master/17.PNG)

At this point I realized that I still had ID included in the attributes and because that is a unique number for each patient I think that might be throwing off my total, so here I am about to rerun all of my data from above to see if the K Neighbors Regressor improves with the removal of ID.  Because the ID is random regardless of the outcome of the tumor, we know the accuracy will increase.

![Image description](https://github.com/JohnRegis/practicum2/blob/master/18.PNG)

![Image description](https://github.com/JohnRegis/practicum2/blob/master/19.PNG)

![Image description](https://github.com/JohnRegis/practicum2/blob/master/20.PNG)

![Image description](https://github.com/JohnRegis/practicum2/blob/master/21.PNG)

![Image description](https://github.com/JohnRegis/practicum2/blob/master/22.PNG)

Rerunning a simple K Neighbors Regressor we have an accuracy score of 0.9210526315789473. This shows how getting rid of useless data truly helps increase the accuracy of the model.  With a Cohens Kappa Score of 0.8355242064764348 which is between 0.81 – 0.99 we can see that this model has a near perfect agreement this shows that this model is completely due to the logic of the model and not based off of chance. This score is much better than when we had the ID column still included.  This shows that the ID column really impacted the model and made the model 'guess' more often than actually predict off of logic.

Next, I did a Random Forest Classifier with an 80% train size and a 20% test size.

![Image description](https://github.com/JohnRegis/practicum2/blob/master/23.PNG)

![Image description](https://github.com/JohnRegis/practicum2/blob/master/24.PNG)

![Image description](https://github.com/JohnRegis/practicum2/blob/master/25.PNG)


The model score I got for this was 0.9473684210526315 which is better than the K Neighbors Regressor.  The  Cohen Kappa Score is 0.8906999041227229 which is between 0.81 – 0.99 so we can see that this model has a near perfect agreement.  This shows that this model is completely due to the logic of the model and not based off of chance.

Here are the features of importance
![Image description](https://github.com/JohnRegis/practicum2/blob/master/26.PNG)

Then I ordered it by most important to least.
![Image description](https://github.com/JohnRegis/practicum2/blob/master/27.PNG)

The last model that I worked on was a decision tree classifier.

![Image description](https://github.com/JohnRegis/practicum2/blob/master/28.PNG)

![Image description](https://github.com/JohnRegis/practicum2/blob/master/29.PNG)

![Image description](https://github.com/JohnRegis/Practicum2/blob/master/30.png)

![Image description](https://github.com/JohnRegis/practicum2/blob/master/31.PNG)

![Image description](https://github.com/JohnRegis/practicum2/blob/master/32.PNG)

The model score I got was 0.9210526315789473 with a Cohen Kappa Score of 0.8386284995281535. which is between 0.81 – 0.99 so we can see that this model has a near perfect agreement.  This shows that this model is completely due to the logic of the model and not based off of chance.

Now I want to try and increase the results of our best model.  From the 3 models that we have run through so far it looks like the Random Forest Classifier provides the best results at 94.7% accuracy with 6 patients being misdiagnosed.  We are going to update the parameters of our Random Forest Classifier.
Here are our current parameters on our Random Forest Classifier.

![Image description](https://github.com/JohnRegis/practicum2/blob/master/33.PNG)

Next, we run through many different parameters and see if this increases the model’s accuracy:

![Image description](https://github.com/JohnRegis/practicum2/blob/master/34.PNG)

![Image description](https://github.com/JohnRegis/practicum2/blob/master/35.PNG)

![Image description](https://github.com/JohnRegis/practicum2/blob/master/36.PNG)

Here we can see that this model still has the same accuracy score, however it has mispredicted different patient outcomes than the first random forest we did.  Since this is dealing with people and medical treatment you have to weigh what is worse telling someone that has cancer that they don't or telling someone who does not have cancer that they do.  My assumption would be telling someone with cancer that they do not have cancer would be worse because they would be losing time that they could be getting treatment because it was misdiagnosed and something that could have been curable now might be terminal.  Going off of that the original random forest was better than this model, even though these have the same accuracy score.

Here we are trying different hyper parameters to see if we can increase the model score.

![Image description](https://github.com/JohnRegis/practicum2/blob/master/37.PNG)

![Image description](https://github.com/JohnRegis/practicum2/blob/master/38.PNG)

![Image description](https://github.com/JohnRegis/practicum2/blob/master/39.PNG)

Here we can see that this model still has the same accuracy score even though it is changed the best parameters used.  Even after changing some of the parameters our second of hyper parameters, we get the same accuracy score as our first attempt of hyper parameters.  It also appears that the confusion Matrix from the first hyper model is the same as the second hyper model.

## Conclusion:

After running 3 different models it turns out that the best mode for predicting if a breast tumor was malignant or benign turns out to the be Random Forest Classifier with an accuracy score of 94.737% with a Cohen Kappa score of 89.07% which is between 0.81 – 0.99 therefor we can see that this model has a near perfect agreement. In this model there were 6 patients that were misdiagnosed from our model. 2 patients were told that their tumors were cancer when they were actually benign and there were 4 patients that were told their tumors were benign when they were in fact cancer.

A nearly 95% accuracy score for most models is pretty good but since this is dealing with medical data this is not nearly good enough.

To try and increase the model’s accuracy score I reran the Random Forest Classifier with sklearn’s Grid Search CV. This ended up changing the predictions but from 2 false positives to only one false positive but increased the false negatives from 4 to 5. Once again in my opinion since this is dealing with people and medical treatment you have to weigh what is worse telling someone that has cancer that they don't or telling someone who does not have cancer that they do. My assumption would be telling someone with cancer that they do not have cancer would be worse because they would be losing time that they could be getting treatment because it was misdiagnosed and something that could have been curable now might be terminal. Going off of that the original random forest was better than this model, even though these have the same accuracy score. I updated the hyper parameters again and the result was the same. The best model that I was able to produce was the random forest.

There is additional work that would need to be done to further perfect this model before it could ever be used in the medical field. The main thing would be to have a larger data set to test on because with how small the data set is, we are limited. I think that there is the potential for addition information of the people come in with the tumors that could be beneficial in better predicting if the tumor is cancer but that would change the dataset overall. The way that the dataset is set up is to solely predict off of the tumor if it is cancer or not but the data set that I am proposing would take the individual into account. I think that a sister data set of this could be useful by including items such as: Age, sex, history of cancer (of any type), history of smoking, history of drug use, etc. Once again that would be a similar dataset but in essence would completely be different from this dataset, however I think that additional attributes could be useful in identifying with better accuracy if a breast tumor was malignant or benign. Due to HIPAA Laws it is hard to have a medical dataset with a good number of detailed attributes because the details cannot in anyway identify who the patient is. I am not saying that there is anything wrong with the HIPAA Laws, I agree with them, I am just stating that the further study model that I am proposing hypothetically would be hard to obtain.


## Work Cited:
Decision Tree Classification in Python. (n.d.). Retrieved from https://www.datacamp.com/community/tutorials/decision-tree-classification-python

Is there a simple way to change a column of yes/no to 1/0 in a Pandas dataframe? (n.d.). Retrieved from 
https://stackoverflow.com/questions/40901770/is-there-a-simple-way-to-change-a-column-of-yes-no-to-1-0-in-a-pandas-dataframe

Koehrsen, W. (2018, January 10). Hyperparameter Tuning the Random Forest in Python. Retrieved from https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74?gi=d6ab536d8deb
 
Magiya, J. (2019, November 23). Pearson Coefficient of Correlation with Python. Retrieved from 
https://levelup.gitconnected.com/pearson-coefficient-of-correlation-using-pandas-ca68ce678c04

numpy.linspace¶. (n.d.). Retrieved from https://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.linspace.html

sklearn.model_selection.GridSearchCV¶. (n.d.). Retrieved from https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html


Stephanie. (2017, October 12). Cohen's Kappa Statistic. Retrieved from https://www.statisticshowto.com/cohens-kappa-statistic/
