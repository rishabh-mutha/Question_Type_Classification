import numpy as np
import pandas as pd
import random
import re
from pickle import load
from pickle import dump
import nltk
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from nltk.stem import SnowballStemmer
from Question_Classification_Model import QuestionClassify
from sklearn.model_selection import train_test_split


def train(filename):
    with open(filename,"r") as text:
        lines=text.readlines()
    lines=[line.replace("\n","") for line in lines]
    lines=[line.split(",,,") for line in lines]
    questions=[line[0] for line in lines]
    question_type=[line[1] for line in lines]
    question_type=[q.strip() for q in question_type]
    questions=[q.strip() for q in questions]
    lines=[[q,w] for (q,w) in zip(questions,question_type)]
    data=pd.DataFrame(lines,columns=["Question","Question_Type"])
    label_dict = {'who': 1, "what": 2, "when": 3, "affirmation": 4, "unknown": 5}

    model = QuestionClassify(label_dict)
    Xtrain, Xtest, ytrain, ytest = train_test_split(data["Question"].values, data["Question_Type"].values,test_size=0.2)
    model.train(Xtrain,ytrain,label_dict)
    print ("Training Accuracy",model.test(Xtrain,ytrain,label_dict) )
    print ("Test Accuracy", model.test(Xtest, ytest,label_dict) )
    

def test_(filename):
    label_dict = {'who': 1, "what": 2, "when": 3, "affirmation": 4, "unknown": 5}
    with open(filename, 'r') as filename:
        lines = filename.readlines()
    test_labels = []
    test_sents = []
    for line in lines:
        line = line.lstrip()
        line = line.rstrip()
        line = line.replace("\n", "")
        words = line.split()

        test_labels.append(words[0])
        test_sents.append(" ".join(words[1:]))

    test_indices = np.random.permutation(len(test_sents))[:200]
    sents=[test_sents[ind] for ind in test_indices]
    labs=[test_labels[ind] for ind in test_indices]
    model=QuestionClassify()
    predictions=model.predict(sents,label_dict)

    sents=np.array([sents]).transpose()

    labs=np.array([labs]).transpose()

    predictions=np.array([predictions]).transpose()

    all_together=np.hstack((sents,labs,predictions))
    df=pd.DataFrame(all_together,columns=["Question","Given Label","Predicted Label"])

    df.to_csv("predictedlabel_file.csv",index=None)

    print ("csv File saved with predicted label.")




train("LabelledData.txt")

test_("test.label.txt")


















