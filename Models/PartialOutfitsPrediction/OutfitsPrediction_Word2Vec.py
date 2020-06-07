import pandas as pd
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
from sklearn.model_selection import train_test_split
import nltk
import multiprocessing
from PartialOutfits_Evaluator import *
from gensim.models import Word2Vec
import random
from gensim.models.doc2vec import TaggedDocument
import numpy as np

class OutfitPrediction_Word2Vec(object):

    def w2v_tokenize_text(self,text):
        tokens = []
        for sent in nltk.sent_tokenize(text, language='english'):
            for word in nltk.word_tokenize(sent, language='english'):
                tokens.append(word)
        return tokens

    def _compute_validation_metrics(self, testing_dataset,  metrics):
        ev = Evaluator_PartialOutfits(testing_dataset, self.k, self.isStructured)

        # Partial Recommendations
        counter = 0
        for _ in testing_dataset.index:
                currentRow = testing_dataset.iloc[counter][0]
                counter += 1
                if self.isStructured:
                    # Structured Entities
                    top_k = self.top_k_recommendations(currentRow[0:len(currentRow)-1],self.methodNumber, self.k)
                    goal = currentRow[-1:]
                else:
                    # Structured Words
                    top_k = self.top_k_recommendations(currentRow[0:-4],self.methodNumber, self.k)
                    goal = currentRow[-4]
                if top_k != -1:
                    ev.add_instance(goal, top_k)

        metrics['hit_ratio'].append(ev.average_hitRatio())
        metrics['precision'].append(ev.average_precision())
        metrics['ndcg'].append(ev.average_ndcg())

        return metrics

    def top_k_recommendations(self, sequence, user_id=None, exclude=None):
        # Recieves a sequence of (id, outfit, style), and produces k recommendations (as a list of outfit items)
        model = Word2Vec.load("word2vec_1.model")
        try:
            f = np.mean(np.array([model.wv[sequence[i]] for i in range(len(sequence))]), axis=0) # average over last window/2 items
            top = model.wv.most_similar(positive=[f],topn=self.k)
        except:
            print("exception")

        return top[:self.k]

    def __init__(self, k, window, learning_rate, dataFile, testSize,size,epochs, min_count, range):
        super(OutfitPrediction_Word2Vec, self).__init__()
        self.k = k
        self.name = 'Word2Vec Partial Outfits Prediction'
        self.counter = 0
        self.dataFile = dataFile
        self.learning_rate = learning_rate
        self.testSize = testSize
        self.size = size
        self.window_size = window
        self.min_count = min_count
        self.epochs = epochs
        self.range = range


        self.metrics = {
            'hit_ratio' : 0,
            'precision': 0,
            'ndcg': 0
        }
        # Read the outfits and style file - pandas
        # col1 - Style , col2 - user name, col3 - outfit
        df_outfits = pd.read_csv(self.dataFile , names={'col1' , 'col2' , 'col3'})

        corpus = []
        for line in df_outfits['col3']:
            words = [x for x in line.split(' ')]
            corpus.append(words)
            corpus.append(line)
        self.corpus = corpus

        train,test = train_test_split(df_outfits,test_size=self.testSize, random_state=42)
        self.train_tokenized = train.apply(lambda r: TaggedDocument(words=self.w2v_tokenize_text(r['col3']), tags=['train']), axis=1)
        self.test_tokenized = test.apply(lambda r: TaggedDocument(words=self.w2v_tokenize_text( r['col3']), tags=['test']), axis=1)

    def shuffle_corpus(self,sentences):
        shuffled = list(sentences)
        random.shuffle(shuffled)
        return shuffled

    def train(self):


        cores = multiprocessing.cpu_count()
        model_dbow = Word2Vec(self.corpus, sg=0,window=self.window_size,size=self.size,
                 min_count= self.min_count,workers=cores,iter=self.epochs,sample=0.01)
        metrics = {name:[] for name in self.metrics.keys()}


        for i in range(self.range):
            model_dbow.train(self.corpus, total_examples=len(self.corpus), epochs=self.epochs)
            model_dbow.save("word2vec_1.model")

        metrics = self._compute_validation_metrics(self.test_tokenized, metrics)
        print("Validation Metrics are:")
        print(metrics)

