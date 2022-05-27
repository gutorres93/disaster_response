import nltk
nltk.download(['punkt', 'wordnet'])

import pandas as pd
import numpy as np
import sys
import re
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, make_scorer, precision_score, accuracy_score, recall_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report
import pickle

def load_data(database_filepath):
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('messages_df', engine)
    df['message'] = df['message'].astype('str')
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    category_names = Y.columns
    
    return X, Y, category_names

def tokenize(text):
    clean_text = re.sub(r"[^a-zA-Z0-9]", " ", text).lower()
    tok = word_tokenize(clean_text)
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(w, pos='v') for w in tok]
        
    return tokens


def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(AdaBoostClassifier()))])
    
    return pipeline

def evaluate_model(model, X_test, Y_test, category_names):

    y_pred = model.predict(X_test)
    y_pred_df = pd.DataFrame(y_pred)
    y_pred_df.columns = category_names
    
    print('Precision:', precision_score(Y_test.values, y_pred, average='micro'))
    print('Recall:', recall_score(Y_test.values, y_pred, average='micro'))
    print('F1 Score:', f1_score(Y_test.values, y_pred, average='micro'))


def save_model(model, model_filepath):
    with open("classifier.pickle", "wb") as file:
        pickle.dump(model, file)

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()