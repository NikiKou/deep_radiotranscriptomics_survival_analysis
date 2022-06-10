import os
import sys
import pandas as pd
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import statistics

from sklearn.feature_selection import SelectFromModel

from sklearn.feature_selection import VarianceThreshold

from sklearn.linear_model import Lasso

from sklearn.preprocessing import StandardScaler

from sklearn.feature_selection import f_classif as fc

from sklearn.feature_selection import SelectKBest as kbest

from sksurv.metrics import concordance_index_censored

from sksurv.svm import FastSurvivalSVM

from sksurv.nonparametric import kaplan_meier_estimator

import scipy.stats as st

from sklearn.model_selection import StratifiedKFold


#load radiomics and transcriptomics data
radiomics_pd = pkl.load(open("features_for_overall_survival_median/radiomics_pandas.pkl", "rb"))
transcriptomics_pd = pkl.load(open("features_for_overall_survival_median/transcriptomics_pandas.pkl","rb"))

#load deep feature
deep_avg = pkl.load(open("features_for_overall_survival_median/deep_features_avg.pkl", "rb"))
deep_max = pkl.load(open("features_for_overall_survival_median/deep_features_max.pkl", "rb"))

#merge deep features vectors (avg and max) to a single feature vector
deep={}
for net in list(deep_avg.keys()):
    deep[net] = deep_avg[net].merge(deep_max[net], left_index = True, right_index = True)


# import labels
with open("features_for_overall_survival_median/days_OS_dictionary.pkl", 'rb') as f:
    OS_days = pkl.load(f)
    OS_days_pd= pd.DataFrame([OS_days.keys(), OS_days.values()]).T
    OS_days_pd.columns= ['Patient', 'Survival_in_days']
    #print(OS_days_pd)
  
with open("features_for_overall_survival_median/binary_OS_dictionary.pkl", 'rb') as f:
    OS_binary = pkl.load(f)
    OS_binary_pd= pd.DataFrame([OS_binary.keys(), OS_binary.values()]).T
    OS_binary_pd.columns= ['Patient', 'Status']
    #print(OS_binary_pd)

final_OS = pd.merge(OS_binary_pd, OS_days_pd, on="Patient", how="left")
#final_OS = final_OS.drop('Patient',1)
final_OSS = final_OS.set_index('Patient')

print("Data cleaning with VarianceThreshold")
thresholder_rad = VarianceThreshold(threshold=0.0)
thresholder_tran = VarianceThreshold(threshold=0.0)
thresholder_deep = VarianceThreshold(threshold=0.0)

## if i want to keep the name of the features
radiomics_selected = thresholder_rad.fit(radiomics_pd)
transcriptomics_selected = thresholder_tran.fit(transcriptomics_pd)
mask_rad = thresholder_rad.get_support()
mask_tran = thresholder_tran.get_support()
radiomics_ = radiomics_pd.loc[:,mask_rad]
transcriptomics_ = transcriptomics_pd.loc[:,mask_tran]
#print(radiomics_)
#print(radiomics_.shape)

print("Z-normalization")
radiomics_transformed = StandardScaler().fit_transform(radiomics_)
transcriptomics_transformed = StandardScaler().fit_transform(transcriptomics_)

# Transform radiomics, transcriptomics data to DataFrame
radiomics = pd.DataFrame(data=radiomics_transformed, index=radiomics_.index, columns=radiomics_.columns)
transcriptomics = pd.DataFrame(data=transcriptomics_transformed,index=transcriptomics_.index, columns=transcriptomics_.columns)

def apply_feature_selection(df, labels, cutoff_pvalue=0.05):
    X=[]
    for key in list(df.index):
        X.append(df.loc[key])
    X = np.array(X)
    y = np.hstack(labels)
    
    selector = kbest(fc, k="all")
    best_features = selector.fit_transform(X, y)
    #print('after kbest:', best_features)
    #print(best_features.shape)
    f_scores, p_values = fc(X, y)
    #print(f_scores)
    #print(p_values)
    critical_value = st.f.ppf(q=1-cutoff_pvalue, dfn=len(np.unique(y))-1, dfd=len(y)-len(np.unique(y)))
    
    best_indices=[]
    for index, p_value in enumerate(p_values):
        if f_scores[index]>critical_value and p_value<cutoff_pvalue:
            best_indices.append(index)
    print("Best ANOVA features:" + str(len(best_indices)))

    if len(best_features)>0:
        best_columns = np.array(list(df.columns))[best_indices]
        best_features = np.array(list(df[best_columns].values))
    else:
        best_columns = np.array(list(df.columns))
        best_features = np.array(list(df.values))

    try:
        sel_ = SelectFromModel(Lasso(alpha=0.01))
        sel_.fit(best_features, y)
        selected_features_bool = sel_.get_support()
        final_selected=[]
        final_features=[]
        for index, feat_id in enumerate(best_columns):
            if selected_features_bool[index]:
                final_selected.append(feat_id)
                #final_features = np.array(list(df[final_selected].values))
        final_selected = np.array(final_selected)
    except:
        print("No features left after Lasso")
        final_selected = best_columns
        
    print("Best Lasso features: "+str(len(final_selected)))
    
    return final_selected

pids = np.array(list(OS_binary.keys()),dtype=str)
f_labels = np.array(list(OS_binary.values()))
sss = StratifiedKFold(n_splits=4,shuffle=True)
kfolds = []
for train_index, test_index in sss.split(pids,f_labels):
    kfolds.append([pids[train_index],pids[test_index]])

for index,split in enumerate(kfolds):
    print(split[0])

results = {}
for model_name in deep.keys():
    print(model_name)
    deep_selected = thresholder_deep.fit(deep[model_name])
    mask_deep = thresholder_deep.get_support()
    deep_ = deep[model_name].loc[:,mask_deep]    
    
    deep_ = StandardScaler().fit_transform(deep[model_name])
    deep_df = pd.DataFrame(data=deep_,index=deep[model_name].index, columns=deep[model_name].columns)
    #print(deep_df.shape)

    #Keep specific patients with known labels
    patients_split = []
    for patients in list(radiomics_pd.index):
        if patients in list(deep_df.index):
            patients_split.append(patients)
    
    deep_final = deep_df.loc[patients_split]
    #print(deep_final.shape)
    
    Concordance_index = []
    for index,split in enumerate(kfolds):
        tr_split = []
        tst_split = []
        for key in list(radiomics.index):
            if key in list(split[0]):
                tr_split.append(key)
            elif key in list(split[1]):
                tst_split.append(key)

        binary_labels = []
        for pid in list(radiomics.loc[tr_split].index):
            if pid in final_OS.values:
                binary_labels.append(OS_binary[pid])

        days_labels = []
        for pid in list(radiomics.loc[tr_split].index):
            if pid in final_OS.values:
                days_labels.append(OS_days[pid])

        try:
            radiomics_feat = apply_feature_selection(radiomics.loc[tr_split], binary_labels)
            transcriptomics_feat = apply_feature_selection(transcriptomics.loc[tr_split], binary_labels)
            deep_feat = apply_feature_selection(deep_final.loc[tr_split], binary_labels)
        except: 
            print('something went wrong')
            continue

        path="results/SVM_rad_trans_deep/"+model_name+"_SVM_nsplit"+str(index)
        os.mkdir(path)
        pd.DataFrame(radiomics_feat).to_csv(path+"/selected_radiomics.csv")
        pd.DataFrame(transcriptomics_feat).to_csv(path+"/selected_transcriptomics.csv")
        pd.DataFrame(deep_feat).to_csv(path+"/selected_deep.csv")

        selected_radiomics = {}
        for key in list(radiomics.index):
            selected_radiomics[key] = radiomics[radiomics_feat].loc[key].to_numpy()

        selected_transcriptomics = {}
        for key in list(transcriptomics.index):
            selected_transcriptomics[key] = transcriptomics[transcriptomics_feat].loc[key].to_numpy()
        #print('selected transcriptomics', type(selected_transcriptomics))

        selected_deep = {}
        for key in list(deep_final.index):
            selected_deep[key] = deep_final[deep_feat].loc[key].to_numpy()
        
        combined_patterns_rad_trans_deep = {}
        for key in list(selected_deep.keys()):
            try:
                combined_patterns_rad_trans_deep[key] = np.concatenate((selected_radiomics[key], selected_transcriptomics[key], selected_deep[key]))
            except:
                print(key)
                continue
        
        x_pd=pd.DataFrame.from_dict(combined_patterns_rad_trans_deep, orient='index')  
        X_train = x_pd.loc[tr_split]
        X_test = x_pd.loc[tst_split]
        y_train = final_OSS.loc[tr_split]
        y_test = final_OSS.loc[tst_split]
        #print(y_test)
    
        # give structure to y
        struct_arr_train = y_train.astype({'Status':'?','Survival_in_days':'<f8'}).dtypes
        y_train_np = np.array([tuple(x) for x in y_train.values], dtype=list(zip(struct_arr_train.index,struct_arr_train)))  
        struct_arr_test = y_test.astype({'Status':'?','Survival_in_days':'<f8'}).dtypes
        y_test_np = np.array([tuple(x) for x in y_test.values], dtype=list(zip(struct_arr_test.index,struct_arr_test)))  
        #print(y_test_np)
        
        estimator = FastSurvivalSVM(alpha=1, max_iter=1000, tol=1e-5, random_state=0)
        estimator.fit(X_train,y_train_np)
        score_ = estimator.score(X_test,y_test_np)
        
        print(score_)
        Concordance_index.append(score_)

        pred = estimator.predict(X_test)
        predictions = np.round(pred, 3)
        pd.DataFrame(predictions).to_csv(path+"/predictions_SVM_rad_trans_deep.csv")
        pd.DataFrame(y_test_np).to_csv(path+"/labels_test_SVM_rad_trans_deep.csv")

    try:
        print('List of possible CI:', Concordance_index)
        print('\nMaximum CI That can be obtained from this model is:',max(Concordance_index))
        print('\nMinimum CI:',min(Concordance_index))
        print('\nMean CI:',statistics.mean(Concordance_index))
        print('\nStandard Deviation is:', statistics.stdev(Concordance_index))
    except:
        print('only one CI points')
        continue
    
    results["rad_trans_deep for model "+model_name] = pd.Series({"Maximum CI":max(Concordance_index),"Minimum CI":min(Concordance_index),"Overall CI":statistics.mean(Concordance_index),"Standard Deviation":statistics.stdev(Concordance_index)})    


final_results = pd.DataFrame.from_dict(results, orient="index")
final_results.to_csv("results/SVM_rad_trans_deep/final_results_SVM_rad_trans_deep.csv") 
