import pandas as pd
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import os

#### Naive Bayes
class StyleClassification_NB(object):
    def __init(self, dataPath):
        self.dataPath = dataPath
        self.testSize = 0.3

    def processModel(self):

        df_outfits = pd.read_csv(self.dataPath, names={'col1', 'col2'})

        X = df_outfits.col2 # outfist
        y = df_outfits.col1 #tags
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.testSize, random_state = 42)

        # Naive Bayes classifier
        # Using pipeline class as a compound classifier
        nb = Pipeline([('vect', CountVectorizer()),
                       ('tfidf', TfidfTransformer()),
                       ('clf', MultinomialNB()),
                      ])
        nb.fit(X_train, y_train)
        y_pred = nb.predict(X_test)
        print(self.dataPath)
        print('accuracy %s' % accuracy_score(y_pred, y_test))
        print(classification_report(y_test, y_pred))

# Iterate over folder of data files that are used in the evaluation
for filename in os.listdir('REMOVED'):
    obj = StyleClassification_NB(filename)
