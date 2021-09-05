
# Hi, I'm Ibrahim! 👋 and this is a
Machine Learning Project on Titanic Survival Prediction and Exploratory Data Analysis
## Dataset
The RMS Titanic was known as the unsinkable ship and was the largest, most luxurious passenger ship of its time. Sadly, the British ocean liner sank on April 15, 1912, killing over 1500 people while just 705 survived.

### Data Set Column Descriptions
- pclass: Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)
- survived: Survival (0 = No; 1 = Yes)
- name: Name
- sex: Sex
- age: Age
- sibsp: Number of siblings/spouses aboard
- parch: Number of parents/children aboard
- fare: Passenger fare (British pound)
- cabin: Cabin number, which looks like ‘C123’ (the letter refers to the deck)
- embarked: Port of embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)
- boat
- body
- home.dest

The full dataset is obtained from [GitHub](https://github.com/jafarijason/my-datascientise-handcode/tree/master/005-datavisualization)
## 🛠 Install Dependencies
    # Pandas and Numpy
    import numpy as np 
    import pandas as pd 

    # Visualization
    import seaborn as sns
    %matplotlib inline
    from matplotlib import pyplot as plt
    from matplotlib import style

    # ML Models
    from sklearn import linear_model
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import Perceptron
    from sklearn.linear_model import SGDClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC, LinearSVC
    from sklearn.naive_bayes import GaussianNB

    # Evaluation
    import re
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import roc_curve
    from sklearn.metrics import precision_recall_curve
    from sklearn.metrics import precision_score, recall_score
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import train_test_split


## Exploratory Data Analysis EDA
- How age and sex affect survival?
- How passenger class affects survival?
- Does having relatives on the ship affect survival?

## Feature Engineering
- Missing Data Handling
    - Filling missing ages 
    - Filling missing embarked
- Create new features 
    - Extract deck number from cabin number
    - Extract titles out of names (Mr, Ms, etc..)
- Convert float "fare" to integer
- Gender Encoding
- Embarked Encoding
- Age Categorization

        df['age'] = df['age'].astype(int)
        df.loc[ df['age'] <= 11, 'age'] = 0
        df.loc[(df['age'] > 11) & (df['age'] <= 18), 'age'] = 1
        df.loc[(df['age'] > 18) & (df['age'] <= 22), 'age'] = 2
        df.loc[(df['age'] > 22) & (df['age'] <= 27), 'age'] = 3
        df.loc[(df['age'] > 27) & (df['age'] <= 33), 'age'] = 4
        df.loc[(df['age'] > 33) & (df['age'] <= 40), 'age'] = 5
        df.loc[(df['age'] > 40) & (df['age'] <= 66), 'age'] = 6
        df.loc[ df['age'] > 66, 'age'] = 6

- Fare Categorization

        df.loc[ df['fare'] <= 7.91, 'fare'] = 0
        df.loc[(df['fare'] > 7.91) & (df['fare'] <= 14.454), 'fare'] = 1
        df.loc[(df['fare'] > 14.454) & (df['fare'] <= 31), 'fare']   = 2
        df.loc[(df['fare'] > 31) & (df['fare'] <= 99), 'fare']   = 3
        df.loc[(df['fare'] > 99) & (df['fare'] <= 250), 'fare']   = 4
        df.loc[ df['fare'] > 250, 'fare'] = 5
        df['fare'] = df['fare'].astype(int)


## Machine Learning Models
**This a comprehensive comparison between 8 ML models**
- Stochastic Gradient Descent
- Random Forest
- Logistic Regression
- K-Nearest Neighbours
- Gaussian Naive Bayes
- Simple Perceptron
- Linear Support Vector Classifier
- Decision Tree Classifier

## Evaluation Metrics
- Training Accuracy
- Confusion Martix
- ROC & AUC 



  
## Authors

- [@96ibman](https://www.github.com/96ibman)

  
## 🚀 About Me
Ibrahim M. Nasser, a Software Engineer, Usability Analyst, 
and a Machine Learning Researcher.


  
## 🔗 Links
[![GS](https://img.shields.io/badge/-Google%20Scholar-blue)](https://scholar.google.com/citations?user=SSCOEdoAAAAJ&hl=en&authuser=2/)

[![linkedin](https://img.shields.io/badge/-Linked%20In-blue)](https://www.linkedin.com/in/ibrahimnasser96/)

[![Kaggle](https://img.shields.io/badge/-Kaggle-blue)](https://www.kaggle.com/ibrahim96/)

  
## Contact

96ibman@gmail.com

  
