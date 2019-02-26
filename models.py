# -*- coding: utf-8 -*-
"""
    Functions to deal with sklearn vectorizers and classifiers 
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn import datasets
import pickle
import os

# Paths to save trained models
MODEL_PATH = {
    'TFIDF_VECTORIZER': 'tfidf_vectorizer.pkl',
    'LOG_REGR_CLASSIFIER': 'log_regr_classifier.pkl',
    'LOG_REGR_DOC2VEC_CLASSIFIER': 'log_regr_doc2vec.pkl',
    'LOG_REGR_POLARITY_CLASSIFIER': 'log_regr_pol.pkl',
    'RF_CLASSIFIER': 'rf_classifier.pkl',
    'LINEARSVC_CLASSIFIER': 'linearsvc_classifier.pkl',
    'SVC_CLASSIFIER': 'svc_classifier.pkl',
    'ADABOOST_CLASSIFIER': 'ada_boost_classifier.pkl',
    'GBC_CLASSIFIER': 'gbc_classifier.pkl'
}

class Model():
    """
       Wrapper to deal with sklearn vectorizers and classifiers.
    """
    
    def tfidf_train(self, x_train, ngram_range=(1,2), max_df=0.9, min_df=5, \
                    max_features=12000, model_file=MODEL_PATH['TFIDF_VECTORIZER']):
        """
            Create and fit sklearn.feature_extraction.text.TfidfVectorizer, write its pickled representation to file.
            
            x_train: an iterable to fit on
            ngram_range: tuple, range of n-grams to be extracted
            max_df: flomax document frequency
            min_df: min document frequency
            max_features: number of terms in the dictionary
            model_file: path to save the model
            
            return: tuple of two items:
                    tf-idf-document-term matrix,
                    TfidfVectorizer                       
        """
        tfidf_vectorizer = TfidfVectorizer(ngram_range=ngram_range, max_df=max_df, \
                                           min_df=min_df, max_features=max_features)
        x_train = tfidf_vectorizer.fit_transform(x_train)
        pickle.dump(tfidf_vectorizer, open(model_file, 'wb'))
        return x_train, tfidf_vectorizer
    
    def tfidf_trans(self, tfidf_vect, x_test):
        """
            Transform documents to tf-idf-document-term matrix.
            
            tfidf_vect: TfidfVectorizer
            x_test: an iterable
            
            return: tf-idf-document-term matrix     
        """        
        return tfidf_vect.transform(x_test)
    
    def tune_params(self, clf, grid_values, x_train, y_train, model_file):
        """
            Apply sklearn.model_selection.GridSearchCV to "clf" estimator, write its pickled representation to file.
            
            clf: sklearn estimator
            grid_values: dictionary with parameters names and values
            
            return: fitted estimator
        """        
        clf_cv = GridSearchCV(clf, grid_values, cv=5, scoring="accuracy", n_jobs=-1)
        clf_cv.fit(x_train, y_train)
        pickle.dump(clf, open(model_file, 'wb'))        
        return clf_cv
        
    def log_regression_train(self, x_train, y_train, model_file=MODEL_PATH['LOG_REGR_CLASSIFIER']):
        """
            Create sklearn.linear_model.LogisticRegression, 
            tune parameters with sklearn.model_selection.GridSearchCV,
            write pickled representation to file.
            
            x_train: training vector
            y_train: target vector
            model_file: path to save the model
            
            return: fitted LogisticRegression classifier 
        """        
        clf = LogisticRegression(random_state=420, solver='liblinear')
        grid_values = {'penalty': ['l1', 'l2'], 
                       'C': [0.01, 1, 10, 100]}
        clf_tuned = self.tune_params(clf, grid_values, x_train, y_train, model_file)
        return clf_tuned
    
    def random_forest_train(self, x_train, y_train, model_file=MODEL_PATH['RF_CLASSIFIER']):
        """
            Create sklearn.ensemble.RandomForestClassifier, 
            tune parameters with sklearn.model_selection.GridSearchCV,
            write pickled representation to file.
            
            x_train: training vector
            y_train: target vector
            model_file: path to save the model
            
            return: fitted LogisticRegression classifier 
        """  
        clf = RandomForestClassifier(random_state=420)
        grid_values = {'n_estimators': [100, 150],
                       'max_features': ['auto', 'log2'],
                       'max_depth' : [2, 4, 5],
                       'criterion' :['gini', 'entropy'],
                       'min_samples_leaf': [1, 2, 3],
                       'min_samples_split': [2, 3]}
        
        clf_tuned = self.tune_params(clf, grid_values, x_train, y_train, model_file)
        return clf_tuned
    
    def linear_svc_train(self, x_train, y_train, model_file=MODEL_PATH['LINEARSVC_CLASSIFIER']):
        """
            Create sklearn.svm.LinearSVC, 
            tune parameters with sklearn.model_selection.GridSearchCV,
            write pickled representation to file.
            
            x_train: training vector
            y_train: target vector
            model_file: path to save the model
            
            return: fitted LogisticRegression classifier 
        """
        clf = LinearSVC(random_state=420, dual=False)
        grid_values = {'penalty': ['l1', 'l2']}
        clf_tuned = self.tune_params(clf, grid_values, x_train, y_train, model_file)
        return clf_tuned
    
    def svc_train(self, x_train, y_train, model_file=MODEL_PATH['LINEARSVC_CLASSIFIER']):
        """
            Create sklearn.svm.SVC, 
            tune parameters with sklearn.model_selection.GridSearchCV,
            write pickled representation to file.
            
            x_train: training vector
            y_train: target vector
            model_file: path to save the model
            
            return: fitted LogisticRegression classifier 
        """
        
        clf = SVC(random_state=420, kernel='linear', probability=True)
        grid_values = {'C': [1, 10]}
        clf_tuned = self.tune_params(clf, grid_values, x_train, y_train, model_file)
        return clf_tuned
    
    def adaboost_train(self, x_train, y_train, model_file=MODEL_PATH["ADABOOST_CLASSIFIER"]):
        """
            Create sklearn.ensemble.AdaBoostClassifier with 
            sklearn.tree.DecisionTreeClassifier as classifier, 
            tune parameters with sklearn.model_selection.GridSearchCV,
            write pickled representation to file.
            
            x_train: training vector
            y_train: target vector
            model_file: path to save the model
            
            return: fitted LogisticRegression classifier 
        """        
        dtc = DecisionTreeClassifier(random_state=123)
        clf = AdaBoostClassifier(base_estimator=dtc, random_state=420)
        grid_values = {'base_estimator__criterion': ['gini', 'entropy'],
                       'base_estimator__max_depth': [2, 4, 5],
                       'base_estimator__max_features': ['auto', 'log2'],
                       'base_estimator__min_samples_leaf': [1, 2, 3],
                       'base_estimator__min_samples_split': [2, 3],
                       'n_estimators' : [50, 80],
                       'learning_rate': [1, 0.1, 0.05]}
        clf_tuned = self.tune_params(clf, grid_values, x_train, y_train, model_file)
        return clf_tuned
    
    def sgd_train(self, x_train, y_train, model_file=MODEL_PATH["ADABOOST_CLASSIFIER"]):
        """
            Create sklearn.ensemble.GradientBoostingClassifier, 
            tune parameters with sklearn.model_selection.GridSearchCV,
            write pickled representation to file.
            
            x_train: training vector
            y_train: target vector
            model_file: path to save the model
            
            return: fitted LogisticRegression classifier 
        """
        clf = GradientBoostingClassifier(random_state=420)
        grid_values = {"loss":["deviance"],
                                    "learning_rate": [1, 0.1, 0.05],
                                    "min_samples_split": [0.1, 0.3],
                                    "min_samples_leaf": [1, 2, 3],
                                    "max_depth":[2, 4, 8],
                                    "max_features":["log2","sqrt"],
                                    "criterion": ["friedman_mse",  "mae"],
                                    "subsample":[0.5, 0.6, 0.8],
                                    "n_estimators": [80, 100, 150]
                                    }
        clf_tuned = self.tune_params(clf, grid_values, x_train, y_train, model_file)
        return clf_tuned
    
    def pred_prob(self, clf, x):
        """
            Predict class probabilities for input samples.
            clf: sklearn estimator
            x: input samples
            
            return: class probabilities for input samples. 
        """
        return clf.predict_proba(x)
    
    def pred_labeles(self, clf, x):
        """
            Predict class label for input samples.
            clf: sklearn estimator
            x: input samples
            
            return: class labels for input samples. 
        """
        return clf.predict(x)
    
    def check_acc(self, clf, x, y, cv=5):
        """
            Evaluate "clf" estimator.
            
            clf: sklean estimator
            x: test samples
            y: true lables for x
            cv: number of folders for cross validation
            
            return: tuple of two elements:
                    mean accuracy on the given data,
                    arrays of scores of the estimator for each run of the cross validation
        """
        clf_scr = clf.score(x, y)
        clf_cross_scr = cross_val_score(clf, x, y, cv=cv)
        print "Mean accuracy: ", clf_scr
        print "Cross validation score: ", clf_cross_scr
        return clf_scr, clf_cross_scr
        
# test
if __name__ == "__main__":
    diabetes = datasets.load_iris()
    X = diabetes.data[:150]
    y = diabetes.target[:150]
    model = Model()
    #clf = model.log_regression_train(X, y)
    #clf = model.random_forest_train(X, y)
    #clf = model.linear_svc_train(X, y)
    #clf = model.svc_train(X, y)
    #clf = model.adaboost_train(X, y)
    clf = model.sgd_train(X, y)
    
    print model.pred_labeles(clf, X)[0:10]
    print model.pred_prob(clf, X)[0:10]
    print model.check_acc(clf, X, y)
    