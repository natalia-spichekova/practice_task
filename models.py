from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC, SVC
import pickle

# Paths to save trained models
MODEL_PATH = {
    'TFIDF_VECTORIZER': 'tfidf_vectorizer.pkl',
    'LOG_REGR_CLASSIFIER': 'log_regr_classifier.pkl',
    'RF_CLASSIFIER': 'rf_classifier.pkl',
    'LINEARSVC_CLASSIFIER': 'linearsvc_classifier.pkl',
    'SVC_CLASSIFIER': 'svc_classifier.pkl',
    'ADABOOST_CLASSIFIER': 'ada_boost_classifier.pkl',
    'GBC_CLASSIFIER': 'gbc_classifier.pkl'
}

class Model():
    
    def tfidf_train(self, x_train, model_file=MODEL_PATH['TFIDF_VECTORIZER']):
        tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,2), max_df=0.9, min_df=5, max_features=12000)
        x_train = tfidf_vectorizer.fit_transform(x_train)
        pickle.dump(tfidf_vectorizer, open(model_file, 'wb'))
        return x_train, tfidf_vectorizer
    
    def tfidf_trans(self, tfidf_vect, x_test):        
        return tfidf_vect.transform(x_test)
    
    def tune_params(self, clf, grid_values, x_train, y_train, model_file):
        clf_cv = GridSearchCV(clf, grid_values, cv=5, scoring="accuracy")
        clf_cv.fit(x_train, y_train)

        pickle.dump(clf, open(model_file, 'wb'))
        
        return clf_cv
        
    def log_regression_train(self, x_train, y_train, model_file=MODEL_PATH['LOG_REGR_CLASSIFIER']):
        
        clf = LogisticRegression(random_state=420, solver='liblinear')
        grid_values = {'penalty': ['l1', 'l2'], 
                       'C': [0.01, 1, 10, 100]}
        
        clf_tuned = self.tune_params(clf, grid_values, x_train, y_train, model_file)
        return clf_tuned
    
    def random_forest_train(self, x_train, y_train, model_file=MODEL_PATH['RF_CLASSIFIER']):
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
        clf = LinearSVC(random_state=420, dual=False)
        
        grid_values = {'penalty': ['l1', 'l2']}
        
        clf_tuned = self.tune_params(clf, grid_values, x_train, y_train, model_file)
        return clf_tuned
    
    def svc_train(self, x_train, y_train, model_file=MODEL_PATH['LINEARSVC_CLASSIFIER']):
        clf = SVC(random_state=420, kernel='linear', probability=True)
        
        grid_values = {'C': [1, 10]}
        
        clf_tuned = self.tune_params(clf, grid_values, x_train, y_train, model_file)
        return clf_tuned
    
    def adaboost_train(self, x_train, y_train, model_file=MODEL_PATH["ADABOOST_CLASSIFIER"]):
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
        return clf.predict_proba(x)
    
    def pred_labeles(self, clf, x):
        return clf.predict(x)
    
    def check_acc(self, clf, x, y, cv=5):
        clf_scr = clf.score(x, y)
        clf_cross_scr = cross_val_score(clf, x, y, cv=cv)
        print "Accuracy: ", clf_scr
        print "Cross validation score: ", clf_cross_scr
        return clf_scr, clf_cross_scr
        
    