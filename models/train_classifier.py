import sys
import sqlite3
import re
import pickle
import os
from datetime import datetime as dt
import time

import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score, make_scorer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from utils import tokenize


def load_data(database_filepath):

    conn = sqlite3.connect(database_filepath)
    df = pd.read_sql('SELECT * FROM dataset', conn)
    X = df['message']
    Y = df.drop(['id', 'original', 'message', 'genre'], axis=1)
    return X, Y


# https://scikit-learn.org/stable/modules/multiclass.html#multiclass
#https://github.com/scikit-learn-contrib/project-template/blob/master/skltemplate/_template.py
def build_model(optimize=False):

    if optimize:
        model = Pipeline([
            ('tfidf', TfidfVectorizer(tokenizer=tokenize, ngram_range=(1, 2))),
            ('clf', MultiOutputClassifier(RandomForestClassifier(class_weight="balanced", n_jobs=-1)))
        ], verbose=True)

        search_space = {
            "tfidf__smooth_idf": Categorical([True, False]),
            "tfidf__min_df": Integer(2, 100),
            "tfidf__norm": Categorical(['l1', 'l2']),
            "tfidf__sublinear_tf": Categorical([True, False]),
            "clf__estimator__max_depth": Integer(3, 15),
            "clf__estimator__max_features": Categorical(['auto', 'sqrt', 'log2']),
            "clf__estimator__min_samples_leaf": Integer(2, 10),
            "clf__estimator__min_samples_split": Integer(2, 10),
            "clf__estimator__n_estimators": Integer(1000, 5000),
            "clf__estimator__class_weight": Categorical(['balanced', 'balanced_subsample'])
        }

        f1_scorer = make_scorer(f1_score, average='weighted')

        # create grid search object
        cv_model = BayesSearchCV(model,
                           search_spaces=search_space,
                           n_iter=50,
                           scoring=f1_scorer,
                           n_jobs=-1,
                           verbose=3,
                           cv=2,
                           return_train_score=True,
                           random_state=42)

        return cv_model

    else:
        model = Pipeline([
            ('tfidf', TfidfVectorizer(tokenizer=tokenize, ngram_range=(1, 2))),
            ('clf', MultiOutputClassifier(RandomForestClassifier(class_weight="balanced", n_jobs=-1,
                                                                 n_estimators=500, max_depth=5)))
        ], verbose=True)

        return model


def evaluate_model(model, X_test, Y_test):
    Y_pred = model.predict(X_test)
    Y_pred = pd.DataFrame(Y_pred, columns=Y_test.columns)

    reports = []
    for label in list(Y_pred):
        y_test = Y_test[label]
        y_pred = Y_pred[label]
        rep = dict()
        rep['feature'] = label
        rep['f1'] = f1_score(y_test, y_pred)
        rep['pr'] = precision_score(y_test, y_pred, zero_division=0)
        rep['rec'] = recall_score(y_test, y_pred)
        rep['acc'] = accuracy_score(y_test, y_pred)

        # Confusion matrix:
        conf = pd.Series(confusion_matrix(y_test, y_pred).ravel(), index=['tn', 'fp', 'fn', 'tp'])
        conf_dict = {ind: conf[ind] for ind in conf.index}
        rep.update(conf_dict)
        reports.append(pd.Series(rep))
        print(pd.Series(rep))

    df_report = np.round(pd.DataFrame(reports).set_index('feature'), 3)
    df_report.insert(0, 'training_timestamp', dt.now().strftime('%Y-%m-%d, %H:%M:%S'))
    conn = sqlite3.connect('../data/DisasterResponse.db')
    #report_filename = 'evaluation_report.csv'
    df_report.to_sql('TrainingEvaluation', conn, if_exists='append')
    # if report_filename in os.listdir(os.getcwd()):
    #     df_report.to_csv(report_filename, mode='a', header=False)
    # else:
    #     df_report.to_csv(report_filename)
    return df_report


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) in [3, 4]:

        database_filepath, model_filepath = sys.argv[1:3]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
        
        print('Building model...')
        if '--optimize' in sys.argv:
            model = build_model(optimize=True)
        else:
            model = build_model(optimize=False)
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl'\
              'if training have to include hyperparameters tuning, then use:'\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl --optimize')


if __name__ == '__main__':
    main()