import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer
from xgboost import XGBClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB


# https://datahub.io/machine-learning/arrhythmia
data = pd.read_csv("arrhythmia_csv.csv")

data = data[ data["class"].isin([1, 2 ,5, 6])]


Y = data["class"]



X = data[['heartrate','chDI_TwaveAmp','chV6_TwaveAmp','chAVR_QRSTA','chV4_TwaveAmp','chV5_QRSTA',
       ]]



"""
X = data.drop(['class'], axis = 1)
"""


preproccessor = make_pipeline(StandardScaler(), SimpleImputer(strategy = 'most_frequent', add_indicator = True))


column_preproccessor = make_column_transformer((preproccessor, X))


myrf_pipeline = make_pipeline(preproccessor, RandomForestClassifier(n_estimators=150,
                                                              random_state=62, min_samples_split = 13, 
                                                              ))
                             
mydc_pipeline = make_pipeline(preproccessor, DecisionTreeClassifier(random_state=60, min_samples_split = 23
                                                              ))

mysvc_pipeline = make_pipeline(preproccessor,  SVC(kernel="linear", C=4))



estimators = [
    ("Random Forest", myrf_pipeline),
    ("Decision Tree", mydc_pipeline),
    ("SVC", mysvc_pipeline),
]

ultimate_classifier = StackingClassifier(estimators=estimators, cv= 5 , passthrough = False, final_estimator=RandomForestClassifier(random_state=40,n_estimators=80, min_samples_split=15, criterion = "log_loss" ))




f1scores =  cross_val_score(ultimate_classifier, X, Y,
                              cv=5,
                              scoring='f1_macro')




recall_scores =  cross_val_score(ultimate_classifier, X, Y,
                              cv=5,
                              scoring='recall_macro')




precision_scores =  cross_val_score(ultimate_classifier, X, Y,
                              cv=5,
                              scoring='precision_macro')




accuracy_scores =  cross_val_score(ultimate_classifier, X, Y,
                              cv=5,
                              scoring='accuracy')

print("f1_macro scores:\n", f1scores.mean())
print("recall scores:\n", recall_scores.mean())
print("precision scores:\n", precision_scores.mean())
print("accuracy scores:\n", accuracy_scores.mean())

