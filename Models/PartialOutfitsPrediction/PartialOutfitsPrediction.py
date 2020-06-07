import pandas as pd
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
from sklearn import utils
from sklearn.model_selection import train_test_split
import nltk
import multiprocessing
from PartialOutfits_Evaluator import *
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

# Partial Outfits Prediction - Doc2Vec
class PartialOutfitPrediction_Doc2Vec(object):

    def tokenize_text(self,text):
        tokens = []
        for sent in nltk.sent_tokenize(str(text), language='english'):
            for word in nltk.word_tokenize(sent, language='english'):
                if len(word) < 2:
                    continue
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

    def top_k_recommendations(self, sequence,  user_id=None, positiveTags=None, negativeTags=None, exclude=None):
        # Recieves a sequence of (id, outfit, style), and produces k recommendations (as a list of outfit items)
        model = Doc2Vec.load("doc2vec_partial.model")
        inferred = model.infer_vector(sequence)
        if self.methodNumber == 1:
            # Similar by vector method - based on the words
            top = model.similar_by_vector(inferred,topn=self.k)
        elif self.methodNumber == 2:
            # Most similarity
            top = model.most_similar(positive=[model.infer_vector(sequence)],topn=self.k)
        elif self.methodNumber == 3:
            # similarity based on the doc
            top = model.docvecs.most_similar(positive=[model.infer_vector(sequence)],topn=self.k)

        return top[:self.k]

    def __init__(self, k, window, learning_rate, dataFile, testSize, dm, vector_size, range, resultsFilePath, isStructured, methodNumber,epochs):
        super(PartialOutfitPrediction_Doc2Vec, self).__init__()
        self.k = k
        self.window = window
        self.learning_rate= learning_rate
        self.dataFile = dataFile
        self.name = 'Partial Outfits Prediction'
        self.testSize = testSize
        self.dm = dm
        self.vector_size = vector_size
        self.range = range
        self.resultsFilePath = resultsFilePath
        self.isStructured = isStructured
        self.methodNumber = methodNumber
        self.epochs = epochs

        self.metrics = {
            'hit_ratio' : 0,
            'precision': 0,
            'ndcg': 0
        }
        # Read the outfits and style file - pandas
        # col1 - Style , col2 - user name, col3 - outfit

        df_outfits = pd.read_csv(self.dataFile, names={'col1','col3'})

        self.dataset = df_outfits
        train, test = train_test_split(df_outfits, test_size=self.testSize, random_state=42)
        self.train_tagged = train.apply(lambda r: TaggedDocument(words=self.tokenize_text(r['col3']), tags=[r.col3]), axis=1)
        self.test_tagged = test.apply(lambda r: TaggedDocument(words=self.tokenize_text( r['col3']), tags=[r.col3]), axis=1)

        corpus = []
        for line in self.dataset['col3']:
            words = [x for x in line.split(' ')]
            corpus.append(words)
        self.corpus = corpus

    def train(self):
        cores = multiprocessing.cpu_count()
        model_dbow = Doc2Vec(dm=self.dm, vector_size=self.vector_size, negative=5, hs=0, min_count=2, sample = 0, workers=cores)
        model_dbow.build_vocab([x for x in tqdm(self.train_tagged.values)])

        metrics = {name:[] for name in self.metrics.keys()}

        for i in range(self.range):
            model_dbow.train(utils.shuffle([x for x in tqdm(self.train_tagged.values)]), total_examples=len(self.train_tagged.values), epochs=self.epochs)
            model_dbow.save("doc2vec_partial.model")

        metrics = self._compute_validation_metrics(self.test_tagged, metrics)
        print("Validation Metrics Values are:")
        print(metrics)