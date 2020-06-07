import pandas as pd
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier
import os

class StyleClassification_SVM(object):
    def __init(self, dataPath):
        self.dataPath = dataPath
        self.testSize = 0.3

    def processModel(self):
        df_outfits = pd.read_csv(self.dataPath, names={'col1', 'col2'})

        X = df_outfits.col2 # outfits
        y = df_outfits.col1 # tags
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= self.testSize, random_state = 42)

        sgd = Pipeline([('vect', CountVectorizer()),
                        ('tfidf', TfidfTransformer()),
                        ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None)),
                       ])
        sgd.fit(X_train, y_train)

        y_pred = sgd.predict(X_test)

        print('accuracy %s' % accuracy_score(y_pred, y_test))
        print(classification_report(y_test, y_pred))

# Iterate over folder of data files that are used in the evaluation
for filename in os.listdir('REMOVED'):
    obj = StyleClassification_SVM(filename)
    obj.processModel()