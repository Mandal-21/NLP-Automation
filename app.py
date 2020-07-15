import streamlit as st
import pandas as pd
import numpy as np
from textblob import TextBlob
import seaborn as sns
from nlp import NLPNaive





# Create space betwwen two context
def space():
    st.markdown("<br>", unsafe_allow_html=True)


# Heading
st.markdown("<h1 style='text-align: center; color: #3f3f44'>NLP - Naive bayes</h1>", unsafe_allow_html=True)
space()
# Sub-Heading
st.markdown("<strong><p style='color: #424874'>1) This project uses Naive Bayes Algorithm</p></strong>", unsafe_allow_html=True)
st.markdown("<strong><p style='color: #424874'>2) You can choose different cleaning process (Stemming, Lemmatizing)</p></strong>", unsafe_allow_html=True)
st.markdown("<strong><p style='color: #424874'>3) Different type of  Metrics formation (Count Vectorizing, TF-IDF)</p></strong>", unsafe_allow_html=True)
st.markdown("<strong><p style='color: #424874'>4) Plotting Sentimental Analysis, Confusion Metrics and Word Cloud</p></strong>", unsafe_allow_html=True)
space()



# Preprocessing data
def preprocess():
    try:
        preprocessing_option = ["Stemming", "Lemmatizing"]
        preprocessor = st.selectbox("Select Preprocessing Technique", preprocessing_option)
        return preprocessor
 
    except Exception as e:
        print("preprocess ERROR : ", e)

# hyperparameters
def hyperparameter():
    try:
        features = ["2500", "3000", "3500", "4000"]
        max_features = st.selectbox("Maximum Features you want to restrict?", features)
        space()
        space()
        ranges = ["1,1", "1,2", "1,3"]
        ngram_range = st.selectbox("Combination of words", ranges).split(',')
        return max_features, ngram_range

    except Exception as e:
        print("hyperparameter ERROR : ", e)




# Creating Bag of Words
def boW():
    try:
        metrics = ["count_vectorizing", "tfidf"]
        bag_of_words = st.selectbox("Bag of Word Technique?", metrics)
        return bag_of_words

    except Exception as e:
        print("boW ERROR : ", e)

# Converting Target categorical variable into Numerical Variable
def y_label():
    try:
        target_variable = ["Yes", "No"]
        y_option = st.selectbox("Do you want to One Hot Encode Target Variable?", target_variable)
        return y_option

    except Exception as e:
        print("y_label ERROR : ", e)


# Main function
def app():

    # Uploading data
    df = st.file_uploader("Upload a Dataset", type=["csv", "txt"], encoding = "ISO-8859-1")
    space()
    if df is not None:
        # Reading data
        data = pd.read_csv(df, encoding = "ISO-8859-1")
        st.dataframe(data.head())
        space()

        text = st.selectbox("Select Text Column", data.columns)
        space()
        target = st.selectbox("Select Target Column", data.columns)
        space()
        # print(type(text), type(target))


        # Reassigning feature to DataFrame
        data = data[[text, target]]

        # Droping NaN values
        data = data.dropna()

        # Initialising class "NLPNaive"
        nlp_model = NLPNaive(data)
        
        # Displaying final DataFrame
        st.markdown("<h4 style='color: #438a5e'>Final Dataset</h4>", unsafe_allow_html=True)
        st.dataframe(data.head())
        space()


        # Calling functions for Preprocessing, Bag of Words, Target variable
        preprocessor = preprocess()
        space()
        space()
        max_features, ngram_range = hyperparameter()
        space()
        space()
        bag_of_words = boW()
        space()
        space()
        y_option = y_label()
        space()
        space()


        # functions
        def metrix(corpus, bag_of_words, max_features, ngram_range):
            try:
                if bag_of_words == "count_vectorizing":
                    X = nlp_model.count_vectorizing(corpus, int(max_features), (int(ngram_range[0]),int(ngram_range[1])))
                    return X

                elif bag_of_words == "tfidf":
                    X = nlp_model.tf_idf(corpus, int(max_features), (int(ngram_range[0]),int(ngram_range[1])))
                    return X

            except Exception as e:
                print("metrix Error : ", e)


        def target_series(y_option, target):
            try:
                if y_option == "Yes":
                    y = nlp_model.y_encoding(target)
                    return y

                elif y_option == "No":
                    y = data[target]
                    return y
            
            except Exception as e:
                print("target_series ERROR : ", e)

        # Plotting functions
        def plot_wordcloud(corpus, y_test, y_pred):
            st.success("Word Cloud")
            wordcloud = nlp_model.word_cloud(corpus)
            st.image(wordcloud)
            accuracy, cm = nlp_model.cm_accuracy(y_test, y_pred)
            st.success(f"Accuracy : {round(accuracy*100, 2)}%")
            st.image(cm)





        

        # sentiments
        def sentimental(text):
            '''Plotting Sentiments'''

            data['sentiments'] = data[text].apply(nlp_model.sentimental_analysis_clean)

            # Sentiments
            def getSubjectivity(text):
                return TextBlob(text).sentiment.subjectivity

            # Create a function to get the polarity
            def getPolarity(text):
                return  TextBlob(text).sentiment.polarity

            def getAnalysis(score):
                if score < 0:
                    return 'Negative'
                elif score == 0:
                    return 'Neutral'
                else:
                    return 'Positive'


            # Create two new columns 'Subjectivity' & 'Polarity'
            data['Subjectivity'] = data['sentiments'].apply(getSubjectivity)
            data['Polarity'] = data['sentiments'].apply(getPolarity)
            data['Analysis'] = data['Polarity'].apply(getAnalysis)

            st.success("Sentiments")

            sns.countplot(x=data["Analysis"],data=data)
            st.pyplot(use_container_width=True)


            
                

        # Model Creation
        if st.button("Analyze"):
            space()
            if preprocessor == "Stemming":
                corpus = nlp_model.stemming(text)
                X = metrix(corpus, bag_of_words, max_features, ngram_range)
                y = target_series(y_option, target)
                X_train, X_test, y_train, y_test = nlp_model.split_data(X, y)
                y_pred = nlp_model.naive_model(X_train, X_test, y_train, y_test)
                sentimental(text)
                plot_wordcloud(corpus, y_test, y_pred)
                




            elif preprocessor == "Lemmatizing":
                corpus = nlp_model.lemmatizing(text)
                X = metrix(corpus, bag_of_words, max_features, ngram_range)
                y = target_series(y_option, target)
                X_train, X_test, y_train, y_test = nlp_model.split_data(X, y)
                y_pred = nlp_model.naive_model(X_train, X_test, y_train, y_test)
                sentimental(text)
                plot_wordcloud(corpus, y_test, y_pred)
                


    st.markdown("<h4 style='text-align: center; color: #3f3f44'>©2020 Amar Mandal</h4>", unsafe_allow_html=True)

        

 
if __name__ == "__main__":
    app()
