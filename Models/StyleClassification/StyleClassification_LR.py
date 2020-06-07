##### Style Classification Experiment ######
##### KTH 2019 - by Shatha Jaradat ######

import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import os

### Logistic Regression - Style Classification Experiment
class StyleClassification_LR(object):

    def __init__(self, dataPath):
        testSize = 0.3
        df_outfits = pd.read_csv(dataPath, names={'col1', 'col2'})

        X = df_outfits.col2 # outfit description
        y = df_outfits.col1 # style tags
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= testSize, random_state = 42)


        logreg = Pipeline([('vect', CountVectorizer()),
                        ('tfidf', TfidfTransformer()),
                        ('clf', LogisticRegression(n_jobs=1, C=1e5)),
                       ])
        logreg.fit(X_train, y_train)

        y_pred = logreg.predict(X_test)

        print("printing results for doc  " + dataPath)
        print('accuracy %s' % accuracy_score(y_pred, y_test))
        print(classification_report(y_test, y_pred))

# Iterate over folder of data files that are used in the evaluation
for filename in os.listdir('REMOVED'):
    obj = StyleClassification_LR(filename)

