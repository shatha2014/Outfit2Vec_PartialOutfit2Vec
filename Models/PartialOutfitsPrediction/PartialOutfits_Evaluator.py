from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import math
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import recall_score, precision_score
from pycm import ConfusionMatrix

'''
Evaluation class - to compute evaluation metrics on tests
It is used by adding a series of instances: pairs of goals and predictions
then metrics can be computed on the ensemble of instances:
average precision, average recall, average ncdg, average hit ratio
Returns the set of correct predictions
'''

class Evaluator_PartialOutfits(object):
    def __init__(self, dataset, k, isStructured):
        self.instances = []
        self.dataset = dataset
        self.k = k
        self.isStructured = isStructured

        self.metrics = {
            'precision': self.average_precision,
            'ndcg': self.average_ndcg,
            'hit_ratio': self.average_hitRatio
        }

    def add_instance(self, goal, predictions):
        self.instances.append([goal, predictions])

    def average_hitRatio(self):
        hitRatio = 0.
        for goal, prediction in self.instances:
            # 1 goal sequence
            # k prediction sequences with their ratings

            # p[0] is the sequence
            # p[1] is the rating of the sequence
            foundHit = False
            lstPrediction = []
            for i, p in enumerate(prediction[:min(len(prediction), self.k)]):
                lstPrediction.append(p[0])
                lstGoal = self.returnListGoalWords(goal)
                if foundHit:
                    break
                if(i == 0):
                    if(len([value for value in lstGoal if value in lstPrediction]) >= round(len(lstGoal) * 0.7)):
                        hitRatio += 1.0
                        foundHit = True
                if(foundHit == False):
                    if(len([value for value in lstGoal if value in lstPrediction]) >= round(len(lstGoal) * 0.7)):
                        hitRatio += 1.0 / float(i)
                        foundHit = True

        return hitRatio / len(self.instances)

    def average_ndcg(self):
        ndcg = 0.
        for goal, prediction in self.instances:
            dcg = 0.
            max_dcg = 0.
            found = False
            if len(prediction) > 0:
                dcg = 0.
                max_dcg = 0.
                # p[0] is the sequence
                # p[1] is the rating of the sequence
                lstPrediction = []
                for i, p in enumerate(prediction[:min(len(prediction), self.k)]):
                    if found:
                        break
                    lstPrediction.append(p[0])
                    lstGoal = self.returnListGoalWords(goal)

                    # formulas as in https://en.wikipedia.org/wiki/Discounted_cumulative_gain
                    # adding 2 as i starts from 0 - division by zero
                    if i < len(goal):
                        max_dcg += float((np.power(2, p[1]) - 1)) / float(np.log2(i + 2))

                    # if the goal matches any of the predictions
                    if self.checkMatchingWithIndex(lstGoal, lstPrediction):
                        dcg += (float(p[1]) / float(np.log2(i + 2)))

                        found = True

                ndcg += dcg / max_dcg

        return ndcg / len(self.instances)

    def average_precision(self):
        avgprecision = 0.0
        for goal, prediction in self.instances:
            lstGoal = self.returnListGoalWords(goal)
            precision_i = 0.0

            numRelevantItems = 0
            numRecommendedItems = 0
            lstPrediction = []

            for i, p in enumerate(prediction[:min(len(prediction), self.k)]):
                lstPrediction.append(p[0])
                print("list of prediction words are ")
                print(lstPrediction)
                if (len([value for value in lstGoal if value in lstPrediction])) >= round(len(lstGoal) * 0.7):
                    numRelevantItems += 1
                    numRecommendedItems += 1
                    found = True
                precision_i += numRelevantItems / float(i + 1)

            avgprecision += precision_i /(numRecommendedItems + 1)
        return float(avgprecision / (len(self.instances)))

    def checkMatchingWithIndex(self, goalSequence, predictionSequence):
        matchFound = False
        if (len([value for value in predictionSequence if value in goalSequence])  >= round(len(goalSequence) * 0.7)):
                matchFound = True

        return matchFound

    def returnListPredictionsWords(self, predictionSequence):
        for word in predictionSequence:
            lstWordsInPrediction = []
            if self.isStructured:
                tempList = []
                for subword in word.split(' '):
                    tempList.append(subword)
                for subword in tempList:
                    for sub in subword.split('_'):
                        lstWordsInPrediction.append(sub)
            else:
                for subword in word.split(' '):
                    lstWordsInPrediction.append(subword)
        return lstWordsInPrediction

    def returnListGoalWords(self, goalSequence):
        lstWordsInGoal = []
        for word in goalSequence:
            if self.isStructured:
                tempList = []
                for subword in word.split(' '):
                    tempList.append(subword)
                for subword in tempList:
                    for sub in subword.split('_'):
                        lstWordsInGoal.append(sub)
            else:
                for subword in word.split(' '):
                    lstWordsInGoal.append(subword)
        return lstWordsInGoal