import nltk
import string
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from textblob.classifiers import NaiveBayesClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, recall_score, precision_score, f1_score, r2_score
from textblob import TextBlob
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from collections import OrderedDict

nltk.download('stopwords', quiet=True)

stopwords = nltk.corpus.stopwords.words('english')

def preprocess_reviews(dataset):
    translator = str.maketrans('', '', string.punctuation)
    return [TextBlob(review.translate(translator)).correct() for review in dataset["Review Body"]]

def get_sentiment_and_counts(dataset=None, str=None):
    sentiment = {}
    for blob in dataset["Review Body"]:
        if not isinstance(blob, TextBlob):
            blob = TextBlob(blob)
        for word in blob.words:
            if word not in stopwords and len(word) > 2:
                if word not in sentiment:
                    sentiment[word] = [blob.sentiment.polarity, 1]
                else:
                    sentiment[word][0] += blob.sentiment.polarity
                    sentiment[word][1] += 1
    return sentiment

def get_weighed_sentiment_counts(sentiment, min_appearences=20):
    weighted_avg_dict = {}
    for word, (sentiment_sum, count) in sentiment.items():
        if count > min_appearences:
            weighted_avg_dict[word] = sentiment_sum / count
    return OrderedDict(sorted(weighted_avg_dict.items(), key=lambda x: x[1], reverse=True))

def plot_sentiment(top_sentiment, bottom_sentiment):
    plt.figure(figsize=(10, 6))
    plt.bar(top_sentiment.keys(), top_sentiment.values(), color='green')
    plt.bar(bottom_sentiment.keys(), bottom_sentiment.values(), color='red')
    plt.title("Sentiment Analysis of Uber Reviews")
    plt.xlabel("Words")
    plt.ylabel("Weighted Average Polarity")
    plt.xticks(rotation=90)
    plt.legend(["Positive", "Negative"])
    plt.show()

def get_metrics(true, predictions):
    accuracy = accuracy_score(true, predictions)
    recall = recall_score(true, predictions)
    precision = precision_score(true, predictions)
    f1 = f1_score(true, predictions)
    return np.array([accuracy, recall, precision, f1])

def print_metrics(true, predictions):
    accuracy, recall, precision, f1 = get_metrics(true, predictions)
    print("Accuracy:", accuracy)
    print("Recall:", recall)
    print("Precision:", precision)
    print("F1:", f1)
    

def plot_metrics(model_results, title="Metrics", ax=None):
    if not ax:
        _, ax = plt.subplots()
    true, predictions = model_results
    disp = ConfusionMatrixDisplay(confusion_matrix(true, predictions))
    disp.plot(ax=ax)
    ax.set_title(title)
    fontoptions = {'fontsize': 7, 'fontfamily': 'sans-serif'}
    ax.text(-0.5, 1.9, "Accuracy: " + str(accuracy_score(true, predictions)), fontoptions)
    ax.text(-0.5, 2.0, "Recall: " + str(recall_score(true, predictions)), fontoptions)
    ax.text(-0.5, 2.1, "Precision: " + str(precision_score(true, predictions)), fontoptions)
    ax.text(-0.5, 2.2, "F1:  " + str(f1_score(true, predictions)), fontoptions)

def perform_SVC(model, exclude, X_train, y_train, X_test, y_test):
    exclude = ~X_train.columns.isin(exclude)    
    model.fit(X_train.loc[:, exclude], y_train)
    predictions = model.predict(X_test.loc[:, exclude])
    return y_test, predictions

def validate_model(model, splits, X, y):
    kf = StratifiedKFold(n_splits=splits)
    total = []
    correct = 0
    num_samples = 0
    for train, test in kf.split(X, y):
        if isinstance(X, pd.DataFrame):
            model.fit(X.iloc[train], y.iloc[train])
            predictions = model.predict(X.iloc[test])
            correct = np.count_nonzero(y.iloc[test] == predictions)
            num_samples = X.iloc[test].shape[0]
            total.append([*get_metrics(y.iloc[test], predictions), correct, num_samples])
        else:
            model.fit(X[train], y[train])
            predictions = model.predict(X[test])
            correct = np.count_nonzero(y[test] == predictions)
            num_samples = X[test].shape[0]
            total.append([*get_metrics(y[test], predictions), correct, num_samples])
    model.fit(X, y)
    total = np.array(total)
    return total.mean(axis=0)


def predict(model, X):
    return model.predict(X)
