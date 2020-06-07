import pandas as pd
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
from gensim.models import Doc2Vec
from sklearn import utils
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from gensim.models.doc2vec import TaggedDocument
import nltk
import multiprocessing
from sklearn.metrics import accuracy_score, classification_report
import os

### Doc2Vec - Paragaraph Vector
class StyleClassification_Doc2Vec(object):

    def __init__(self, dataPath):
        self.dataPath = dataPath
        self.testSize = 0.3
        self.dm = 0 #1
        self.vector_size = 300 #200
        self.range = 20 #30
        self.epochs = 20

    def processModel(self):
        # Read the outfits and style file - pandas
        df_outfits = pd.read_csv(self.dataPath, names={'col1', 'col2'})

        # print shape of data
        #print(df_outfits.shape)

        #count of styles
        #cnt_styles = df_outfits['col1'].value_counts()

        train, test = train_test_split(df_outfits, test_size=self.testSize, random_state=42)
        train_tagged = train.apply(
            lambda r: TaggedDocument(words=tokenize_text(r['col2']), tags=[r.col1]), axis=1)
        test_tagged = test.apply(
            lambda r: TaggedDocument(words=tokenize_text(r['col2']), tags=[r.col1]), axis=1)

        cores = multiprocessing.cpu_count()
        model_dbow = Doc2Vec(dm=self.dm, vector_size=self.vector_size, negative=5, hs=0, min_count=2, sample = 0, workers=cores)
        model_dbow.build_vocab([x for x in tqdm(train_tagged.values)])


        for epoch in range(self.range): #was 100
            model_dbow.train(utils.shuffle([x for x in tqdm(train_tagged.values)]), total_examples=len(train_tagged.values), epochs=self.epochs)
            model_dbow.alpha -= 0.002
            model_dbow.min_alpha = model_dbow.alpha

        y_train, X_train = vec_for_learning(model_dbow, train_tagged)
        y_test, X_test = vec_for_learning(model_dbow, test_tagged)
        logreg = LogisticRegression(n_jobs=1, C=1e5)
        logreg.fit(X_train, y_train)
        y_pred = logreg.predict(X_test)
        print("printing results of tagged document #  " + self.dataPath)
        print('accuracy %s' % accuracy_score(y_test, y_pred))
        print(classification_report(y_test, y_pred))

    def tokenize_text(self,text):
        tokens = []
        for sent in nltk.sent_tokenize(text):
            for word in nltk.word_tokenize(sent):
                if len(word) < 2:
                    continue
                tokens.append(word.lower())
        return tokens

    def vec_for_learning(self,model, tagged_docs):
        sents = tagged_docs.values
        # steps is alias for epochs
        targets, regressors = zip(*[(doc.tags[0], model.infer_vector(doc.words, steps=20)) for doc in sents])
        return targets,regressors

# Iterate over folder of data files that are used in the evaluation
for filename in os.listdir('REMOVED'):
    obj = StyleClassification_Doc2Vec(filename)
    obj.processModel()

