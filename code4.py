import numpy as np 
import pandas as pd   
### preprocessing data 
from sklearn.preprocessing import LabelEncoder 
#### spliting data
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
## mdoels
from xgboost import XGBClassifier 
from sklearn.ensemble import VotingClassifier 
from sklearn.ensemble import RandomForestClassifier 
#####
from sklearn.metrics import roc_auc_score 


# preparing algorithms to use it
SKF_model = StratifiedKFold(n_splits=5)
XGB_model = XGBClassifier(n_estimators=100,learning_rate=0.07,booster='gbtree',gamma=0.5 ,
                          reg_alpha=0.5 , reg_lambda=0.5,base_score=0.2) 
RF_model = RandomForestClassifier(n_estimators=120 , criterion='gini' ,
                                  min_samples_split=1.0 ,min_samples_leaf=0.5 , max_leaf_nodes=4) 
VC_model = VotingClassifier(estimators=[('XGB' , XGB_model) , ("Rf" , RF_model)] ,  voting='hard') 



test_list=[]
for count , (train_idx ,test_idx) in enumerate(SKF_model.split(X,y)):
    X_train = X.iloc[train_idx] 
    X_valid = X.iloc[test_idx] 
    y_train = y.iloc[train_idx]
    y_valid = y.iloc[test_idx] 
    print("*************fold(" , count+1 , ")***************")
    VC_model.fit(X_train , y_train)
    y_predict = VC_model.predict(X_valid) 
    test_predict = VC_model.predict(X_test) 
    score =roc_auc_score(y_valid , y_predict)
    test_list.append(test_predict)
    print('score------>' , score)
    print("\n")
    
    
    
    
# show test predict
test_predict=test_list[3]
test_predict[:20]














