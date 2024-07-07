# Text Category Assignment

AUEB Informatics

Project 2 for the course "Artificial Intelligence" 5th semester 2023 - 2024

*Avrabos Georgios - p3210001*<br>
*Gasparhs Rhgos - p3210174*<br>
*Griva Aggelikh Aikaterinh - p3210041*<br><br>

## Project Description

In this project we have implemented 3 machine learning algorithms to classify given text into two categories: positive/negative. The algorithms are the following:

**- Naive Bayes**<br>
**- Random Forest**<br>
**- Adaboost**

The project is split into 3 phases: 1) First create the algorithms from scratch, using only numpy and basic computational libraries. 2) Use already implemented algorithms from libraries such as Scikit-learn, Keras etc. and compare the results with the ones of the first phase. 3) Create a MLP with sliding window or RNN.

For our data we use the "IMDB Dataset" (or "Large Movie Review Dataset") which can be found here https://keras.io/api/datasets/imdb. Based on the documentation we create our dataset and use it for the training and testing of the algorithms.

For our vocabulary we search for a range of words that skips very common words which will not help the algorithms determine much and we also want to skip very rare words. For example the word "the" is the most used word in the english language. It would be foolish if we allowed our algorithms to examine this word and account it for their final decision, since this word can be equally found in both positive and negative sentences. On the other hand the word "conundrum" for instance, will be of such low occurance that again, it will not aid the algorithms for an accurate final decision.

## Naive Bayes

### Accuracy and precision diagrams
<p align=center>Our Bayes</p>

![bayes_acc_pre](images/bayes/bayes_a_p.png)
<p align=center>Scikit-learn's Bayes</p>

![bayesSK_acc_pre](images/bayes/bayesSK_a_p.png)
### Recall and F1 diagrams
<p align=center>Our Bayes</p>

![bayes_rec_f1](images/bayes/bayes_a_p.png)
<p align=center>Scikit-learn's Bayes</p>

![bayesSK_rec_f1](images/bayes/bayesSK_r_f.png)

*Credit: Griva Aggelikh Aikaterinh*

## Random Forest

### Accuracy
![rf_acc](images/randomForest/a.png)
### Precision
![rf_pre](images/randomForest/p.png)
### Recall
![rf_rec](images/randomForest/r.png)
### F1
![rf_f1](images/randomForest/f.png)

*Credit: Gasparhs Rhgos*

## Adaboost

#### [Adaboost's implementation can be found here.](Adaboost.ipynb)

### Accuracy
![ada_acc](images/Adaboost/a.png)
### Precision
![ada_pre](images/Adaboost/p.png)
### Recall
![ada_rec](images/Adaboost/r.png)
### F1
![ada_f1](images/Adaboost/f.png)

*Credit: Avrabos Georgios*