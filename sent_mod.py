#Use pickled classes

import nltk
import random
import codecs
import pickle
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from nltk.stem import PorterStemmer
from nltk.metrics import   ConfusionMatrix
from nltk.classify import ClassifierI
from statistics import mode


class Classifier_Aggregator(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self,features):
        aggregate = []
        for c in self._classifiers:
            r = c.classify(features)
            aggregate.append(r)
        return mode(aggregate)

    def confidence(self,features):
        aggregate = []
        for c in self._classifiers:
            r = c.classify(features)
            aggregate.append(r)

        result = aggregate.count(mode(aggregate))
        conf = result / len(aggregate)
        return conf

#Import pickled stuff

document_f = open("document.pickle", "rb")
document = pickle.load(document_f)
document_f.close()


word_features_f = open("word_features5k.pickle","rb")
word_features = pickle.load(word_features_f)
word_features_f.close()





#Feature extractor -- where features are unigrams
def feature_extractor(document):
    words = word_tokenize(document)
    features = {}
    for word in word_features:
        features[word] = (word in words)

    return features

featuresets_f = open("featuresets.pickle","rb")
featuresets = pickle.load(featuresets_f)
featuresets_f.close()

random.shuffle(featuresets)
print("Feature set length: ", len(featuresets))
random.shuffle(featuresets)

training_set, dev_set, testing_set = featuresets[1000:],featuresets[500:1000],featuresets[:500]



#Naive Bayes
open_NB = open("naivebayes5k.pickle", "rb")
NB_classifier = pickle.load(open_NB)
open_NB.close()

#print("Original Naive Bayes Algo accuracy percent:", (nltk.classify.accuracy(NB_classifier, dev_set))*100)


#Logistic Regression
open_LR = open("logisticregression5k.pickle", "rb")
LR_classifier = pickle.load(open_LR)
open_LR.close()
#print("Logistic Regression classifier accuracy percent:", (nltk.classify.accuracy(LR_classifier, dev_set))*100)



#Linear SVM
open_LinearSVM = open("linearsvm5k.pickle", "rb")
LinearSVM_classifier = pickle.load(open_LinearSVM)
open_LinearSVM.close()

#print("Linear SVM classifier accuracy percent:", (nltk.classify.accuracy(LinearSVM_classifier, dev_set))*100)


aggregate_result = Classifier_Aggregator(NB_classifier,LR_classifier,LinearSVM_classifier)

def sentiment(text):
    feats = feature_extractor(text)
    sent = aggregate_result.classify(feats)
    confidence = aggregate_result.confidence(feats)
    return sent,confidence







###########
#Confusion Matrix
#############
#As a form of error analysis for any of the classifiers 
# test,gold=[],[]
# for i in range(len(testing_set)):
#     test.append(LinearSVM_classifier.classify(testing_set[i][0]))
#     gold.append(testing_set[i][1])
# CM = nltk.ConfusionMatrix(gold,test)
# print("Confusion Matrix:\n", CM)
