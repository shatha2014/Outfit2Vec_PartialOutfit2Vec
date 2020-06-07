##### Style Classification Experiment ######

from gensim.models import Word2Vec
from gensim import matutils
import nltk
import pandas as pd
import numpy as np
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import random
import os

# Code related to Word Averaging:
# https://towardsdatascience.com/multi-class-text-classification-model-comparison-and-selection-5eb066197568

# To use word2vec for documents, each word is represented by a word
# embedding vector and then use some "Add" methods like the word averaging
# in order to get a good representation for the document

class StyleClassification_Word2Vec(object):
    def __init__(self,dataPath):
        self.dataPath  = dataPath
        self.size = 200
        self.window_size = 10 # was:
        self.epochs = 20 # epochs were 100
        self.min_count = 2
        self.workers = 4
        self.sg = 0 #1
        self.range = 10 #was 20
        self.test_size = 0.3

    def processModel(self):
        df_outfits = pd.read_csv(self.dataPath, names={'col1', 'col2'})

        corpus = []
        for line in df_outfits['col2']:
            words = [x for x in line.split(' ')]
            corpus.append(words)
        for line in df_outfits['col1']:
            words = [x for x in line.split(' ')]
            corpus.append(words)

        num_of_sentences = len(corpus)
        num_of_words = 0
        for line in corpus:
            num_of_words += len(line)

        print('Num of sentences - %s'%(num_of_sentences))
        print('Num of words - %s'%(num_of_words))

        # train word2vec model using gensim
        model = Word2Vec(corpus, sg=self.sg,window= self.window_size,size=self.size,
                         min_count= self.min_count,workers=self.workers,iter=self.epochs,sample=0.01)
        model.build_vocab(sentences=self.shuffle_corpus(corpus),update=True)

        for i in range(10): #self.range
            model.train(sentences=self.shuffle_corpus(corpus),epochs=self.epochs,total_examples=model.corpus_count)

        train, test = train_test_split(df_outfits, test_size=self.test_size, random_state = 42)

        train_tokenized = train.apply(lambda k: self.w2v_tokenize_text(k['col2']),axis=1).values
        test_tokenized = test.apply(lambda r: self.w2v_tokenize_text(r['col2']), axis=1).values

        X_train_word_average = self.word_averaging_list(model.wv,train_tokenized)
        X_test_word_average = self.word_averaging_list(model.wv,test_tokenized)

        logreg = LogisticRegression(n_jobs=1, C=1e5)
        logreg = logreg.fit(X_train_word_average, train['col1'])
        y_pred = logreg.predict(X_test_word_average)
        print('accuracy %s' % accuracy_score(y_pred, test.col1))
        print(classification_report(test.col1, y_pred))

    def shuffle_corpus(self,sentences):
            shuffled = list(sentences)
            random.shuffle(shuffled)
            return shuffled

    # Taken from above reference

    def word_averaging(self,wv, words):
        all_words, mean = set(), []

        for word in words:

            if isinstance(word, np.ndarray):
                mean.append(word)
            elif word in wv.vocab:
                mean.append(wv.syn0[wv.vocab[word].index])
                all_words.add(wv.vocab[word].index)
        if not mean:
            return np.zeros(wv.vector_size,)

        mean = matutils.unitvec(np.array(mean).mean(axis=0)).astype(np.float32)
        return mean

    # Taken from above reference
    def  word_averaging_list(self,wv, text_list):
        return np.vstack([self.word_averaging(wv, post) for post in text_list ])

    # Taken from above reference
    def w2v_tokenize_text(self,text):
        tokens = []
        for sent in nltk.sent_tokenize(text, language='english'):
            for word in nltk.word_tokenize(sent, language='english'):
                if len(word) < 2:
                    continue
                tokens.append(word)
        return tokens


# Iterate over folder of data files that are used in the evaluation
for filename in os.listdir('REMOVED'):
    obj = StyleClassification_Word2Vec(filename)
    obj.processModel()
