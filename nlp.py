import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

# !pip install -q scikit-plot
import scikitplot as skplt

# !pip install wordcloud
from wordcloud import WordCloud
from PIL import Image
from textblob import TextBlob


class NLPNaive:
    def __init__(self, data):
        self.data = data


    def stemming(self, column_name):
        '''Cleaning data using re, stemming and stopwords'''
        try:
            corpus = []
            stemming = PorterStemmer()
            for i in range(len(self.data)):
                tweet = re.sub('[^a-zA-Z]', " ", self.data[column_name][i])
                tweet = re.sub('http', "", tweet)
                tweet = re.sub('co', "", tweet)
                tweet = re.sub('amp', "", tweet)
                tweet = re.sub('new', "", tweet)
                tweet = re.sub('one', "", tweet)
                tweet = tweet.lower()
                tweet = tweet.split()
                tweet = [stemming.stem(word) for word in tweet if word not in set(stopwords.words("english"))]
                tweet = " ".join(tweet)
                corpus.append(tweet)
            
        except Exception as e:
            print("stemming ERROR : ",e)
        
        else:
            # print("Cleaning was sucessful")
            return corpus


    def lemmatizing(self, column_name):
        '''Cleaning data using re, Lemmatization and stopwords'''
        try:
            corpus = []
            lemmatize = WordNetLemmatizer()
            for i in range(len(self.data)):
                tweet = re.sub('[^a-zA-Z]', " ", self.data[column_name][i])
                tweet = re.sub('http', "", tweet)
                tweet = re.sub('co', "", tweet)
                tweet = re.sub('amp', "", tweet)
                tweet = re.sub('new', "", tweet)
                tweet = re.sub('one', "", tweet)
                tweet = tweet.lower()
                tweet = tweet.split()
                tweet = [lemmatize.lemmatize(word) for word in tweet if word not in set(stopwords.words("english"))]
                tweet = " ".join(tweet)
                corpus.append(tweet)
            
        except Exception as e:
            print("Lemmatizing ERROR : ",e)
    
        else:
            # print("Cleaning was sucessful")
            return corpus


    def count_vectorizing(self, corpus, max_features = 3000, ngram_range=(1,2)):
        '''Creating Bag of Words using CountVectorizer'''
        try:
            cv = CountVectorizer(max_features=max_features, ngram_range=ngram_range)
            X = cv.fit_transform(corpus).toarray()
        
        except Exception as e:
            print("count_vectorizing ERROR : ",e)
        
        else:
            # print("Bag of Words created successfully")
            return X



    def tf_idf(self, corpus, max_features = 3000, ngram_range=(1,2)):
        '''Creating Bag of Words using TfidfVectorizer'''
        try:
            tfidf = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
            X = tfidf.fit_transform(corpus).toarray()
        
        except Exception as e:
            print("tf_idf ERROR : ",e)
        
        else:
            # print("Bag of Words created successfully")
            return X


    def y_encoding(self, target_label):
        """One Hot Encoding if target variable are not in form of 1s and 0s"""
        try:
            y = pd.get_dummies(self.data[target_label], drop_first = True)

        except Exception as e:
            print("y_encoding ERROR : ", e)

        else:
            # print("y encoded sucessfully")
            return y
    


    def split_data(self, X, y, test_size = 0.25, random_state = 0):
        '''Splitting data into train and test set'''
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = random_state)
        
        except Exception as e:
            print("split_data ERROR : ",e)

        else:
            # print("Successful Splitting")
            return X_train, X_test, y_train, y_test


    def naive_model(self, X_train, X_test, y_train, y_test):
        '''Prediction of model using naive_bayes'''
        try:
            naive = MultinomialNB()
            naive.fit(X_train , y_train)

            y_pred = naive.predict(X_test)
        
        except Exception as e:
            print("naive_model ERROR : ", e)
            
        else:
            # print("Naive Bayes Model built successfully")
            return y_pred


    def cm_accuracy(self, y_test, y_pred):
        '''Performace Metrics'''
        try:

            skplt.metrics.plot_confusion_matrix(y_test, 
                                                y_pred,
                                                figsize=(7,7))
            plt.savefig('CM.jpg')
            img_cm= Image.open("CM.jpg")
            accuracy = accuracy_score(y_test, y_pred)
        
        except Exception as e:
            print("cm_accuracy ERROR : ", e)

        else:
            # print("cm_accuracy plotted successfully")
            return accuracy, img_cm


    def word_cloud(self, corpus):
        '''Generating Word Cloud'''
        try:
            wordcloud = WordCloud(
                        background_color='white',
                        width=720,
                        height=500,
                        ).generate(" ".join(corpus))
            plt.imshow(wordcloud, interpolation="bilinear")
            plt.axis('off')
            plt.savefig('WC.jpg')
            img= Image.open("WC.jpg") 
            

        except Exception as e:
            print("word_cloud ERROR : ", e)

        else:
            # print("word cloud plotted")
            return img



    def sentimental_analysis_clean(self, text):
        try:
            text = re.sub('http', "", text)
            text = re.sub('co', "", text)
            text = re.sub('amp', "", text)
            text = re.sub('new', "", text)
            text = re.sub('one', "", text)
            text = re.sub('@[A-Za-z0–9]+', '', text)
            text = re.sub('#', '', text)
            text = re.sub('RT[\s]+', '', text)
            text = re.sub('https?:\/\/\S+', '', text)
        
            return text

        except Exception as e:
            print("sentimental_analysis_clean ERROR : ", e)
