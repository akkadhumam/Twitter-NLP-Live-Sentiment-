#Original set -- save pickles for use in later version

import nltk
import random
import codecs
import pickle
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

#The entire document

###########
positive = codecs.open("rt-polarity.pos","r",encoding="latin2").read()
negative = codecs.open("rt-polarity.neg","r",encoding="latin2").read()


document = []



for l in positive.split('\n'):
    document.append((l,"pos"))

for l in negative.split('\n'):
    document.append((l,"neg"))

#Pickle
save_document = open("document.pickle","wb")
pickle.dump(document, save_document)
save_document.close()


#Words

###############

words = []

positive_words = word_tokenize(positive)
negative_words = word_tokenize(negative)

for word in (positive_words):
    words.append(word.lower())

for word in (negative_words):
    words.append(word.lower())

all_words = nltk.FreqDist(words)

#Feature extraction
#Reminder: re-run 2nd version with scikit-learn's feature extraction module

##################


#Taking the top 25% occuring words
word_features = list(all_words.keys())[:(len(words)//4)]

#pickle it
save_word_features = open("word_features5k.pickle","wb")
pickle.dump(word_features, save_word_features)
save_word_features.close()

# wordfeature_f = open("word_features5k.pickle","rb")
# word_features = pickle.load(wordfeature_f)
# wordfeature_f.close()

#Feature extractor
def feature_extractor(document):
    words = word_tokenize(document) #or try: words= set(document)
    features = {}
    for word in word_features:
        features[word] = (word in words)

    return features


##
#Producing our sets
# featuresets = [(feature_extractor(n),sent) for (n,sent) in document]
# random.shuffle(featuresets)
# print(len(featuresets))

#Pickling featuresets for time efficiency
# save_featuresets = open("featuresets.pickle","wb")
# pickle.dump(featuresets,save_featuresets)
# save_featuresets.close()

featuresets_f = open("featuresets.pickle","rb")
featuresets = pickle.load(featuresets_f)
featuresets_f.close()

random.shuffle(featuresets)
print(len(featuresets))

training_set, dev_set, testing_set = featuresets[1000:],featuresets[500:1000],featuresets[:500]

#classifiers
#########

#Naive NaiveBayes
############
NB_classifier=  nltk.NaiveBayesClassifier.train(training_set)
print("Original Naive Bayes Algo accuracy percent:", (nltk.classify.accuracy(NB_classifier, testing_set))*100)
NB_classifier.show_most_informative_features(15)

#Pickle it
save_NB_classifier = open("naivebayes5k.pickle","wb")
pickle.dump(NB_classifier, save_NB_classifier)
save_NB_classifier.close()



###############

#Logistic Regression
################
LR_classifier = SklearnClassifier(LogisticRegression())
LR_classifier.train(training_set)
print("Logistic Regression classifier accuracy percent:", (nltk.classify.accuracy(NB_classifier, testing_set))*100)

#Pickle it
save_LR_classifier = open("logisticregression5k.pickle","wb")
pickle.dump(LR_classifier, save_LR_classifier)
save_LR_classifier.close()

####################

#Linear SVM
#################
LinearSVM_classifier = SklearnClassifier(LinearSVC())
LinearSVM_classifier.train(training_set)
print("Linear SVM classifier accuracy percent:", (nltk.classify.accuracy(LinearSVM_classifier, testing_set))*100)

#Pickle it
save_LinearSVM_classifier = open("linearsvm5k.pickle","wb")
pickle.dump(LinearSVM_classifier, save_LinearSVM_classifier)
save_LinearSVM_classifier.close()
