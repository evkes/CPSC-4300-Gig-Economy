import nltk
import string
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from textblob.classifiers import NaiveBayesClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, recall_score, precision_score, f1_score
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

def print_metrics(true, predictions):
    print("Accuracy:", accuracy_score(true, predictions))
    print("Recall:", recall_score(true, predictions))
    print("Precision:", precision_score(true, predictions))
    print("F1:", f1_score(true, predictions))

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

def predict(model, X):
    return model.predict(X)
