
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import BatchNormalization
import numpy as np
import math
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense,Flatten
import matplotlib.pyplot as plt
import shap
#import graphviz
SubjectInfo = pd.read_csv("/project/ssverma_shared/projects/Periodontitis/Periodontitis/ads_subj_lelvel_metadata_csv.csv",dtype= {"SUBJID":"string"},index_col="SUBJID")
SubjectInfo = SubjectInfo[SubjectInfo["COMPLEFL"]== "Y"] #remove subjects who were marked as not having completed the study
#remove dates and communication
SubjectInfo.drop(["DOB","BASEDT","INFCDT","ELIGFL","ELIGDT","NOCONT","NOCNTOTH","SCRDT","MO2DT","MO4DT","MO6DT","MO8DT","MO10DT","MO12DT","PTPVDT","PTSVDT","PMO3DT","PMO6DT","COMPLEDT","DSREASCD","DSREAS","DSSPEC","LASTDT","DSCOMM"],axis=1,inplace=True)
SubjectInfo.dropna

PRSScores  = pd.read_csv("/project/ssverma_shared/projects/Periodontitis/Periodontitis/NoLabels_PRScs-Table.csv",dtype= {"Subject ID":"string"},index_col="Subject ID")
PRSScores = pd.DataFrame(PRSScores[['PRScs_SCORE']])
PRSScores["PRSSTAN"] = PRSScores.apply(lambda x:(x['PRScs_SCORE']- PRSScores['PRScs_SCORE'].mean()) / PRSScores['PRScs_SCORE'].std(),axis=1)

#SerumInfo = pd.read_csv("/Users/lannacaruth/PycharmProjects/pythonProject1/Periodontitis/serum_merged copy.csv",dtype= {"Subject_ID":"string"},index_col='Subject_ID')
#SerumInfo = SerumInfo[SerumInfo["Visit_number"]=="Baseline"]
#SerumInfo.drop(['PDCLASS', 'progclass', 'Visit_number'],axis=1,inplace=True) #"MMP_9__14_","Human_serum_MPO__53_"])
#SerumInfo.columns

SalivaInfo = pd.read_csv("/project/ssverma_shared/projects/Periodontitis/Periodontitis/saliva_merged_HP.csv.gz",dtype= {"Subject_ID":"string"},index_col='Subject_ID')
SalivaInfo = SalivaInfo[SalivaInfo["Visit"]=="Baseline"]
SalivaInfo.drop(["PDCLASS","progclass","Visit","Tag"],axis=1,inplace=True)
SalivaInfo.dropna

SiteLevelInfo = pd.read_csv("/project/ssverma_shared/projects/Periodontitis/Periodontitis/adpd_periodontaldata.csv.gz",dtype= {"SUBJID":"string"},index_col='SUBJID') #one record per subject per visit per tooth per tooth site
SiteLevelInfo = SiteLevelInfo[SiteLevelInfo["VISITN"]==0]
SiteLevelInfo = SiteLevelInfo[["ID","VISITN","PDEP","TOMISSFN","ABCESSFN","TOQT","BOPFN"]] 
toothnumbers = SiteLevelInfo["TOQT"].unique()

AdditionalSiteInfo = pd.read_csv("/project/ssverma_shared/projects/Periodontitis/Periodontitis/adpd_AI_summary.csv.gz",dtype= {"SUBJID":"string"},index_col='SUBJID') 
AdditionalSiteInfo = AdditionalSiteInfo[AdditionalSiteInfo["VISITN"]==0]


#MouthLevelInfoIncludeClinStats = SiteLevelInfoIncludeClinStats[["SUBJID","ID"]]
SubjectProgressionInfo = pd.read_csv("/project/ssverma_shared/projects/Periodontitis/Periodontitis/Subj_level_PDClass_ProgressionClass.csv",dtype={"subjid":"string"},index_col="subjid")
#remove index and initial classification columns
SubjectProgressionInfo.drop(["Obs","PDCLASS"],axis=1,inplace=True)
#match naming and type converntions in subject level data frame
SubjectProgressionInfo.rename(columns={"progclass":"PROGCLASS"},inplace=True)


AdditionalSiteInfo = AdditionalSiteInfo[~AdditionalSiteInfo.index.duplicated(keep='first')]
FinalClinData = pd.merge(SubjectInfo,AdditionalSiteInfo["PDCD"],how="inner",left_index=True,right_index=True)
FinalClinData.drop(columns="DEMOCOMM",inplace=True)
FinalClinData = pd.merge(FinalClinData,PRSScores,how="inner",left_index=True,right_index=True)
FinalClinData = pd.merge(FinalClinData,SalivaInfo,how="inner",left_index=True,right_index=True)
FinalClinData = pd.merge(FinalClinData,SubjectProgressionInfo,how="inner",left_index=True,right_index=True)
#print(FinalClinData.isna().sum())


#creating mouth level stats
#abcessdf = SiteLevelInfoIncludeClinStats.loc[(SiteLevelInfoIncludeClinStats["SUBJID"]== ID) & (SiteLevelInfoIncludeClinStats["TOQT"] == tnum)]
#if 1 in abcessdf.ABCESSFN.values:
           #abcesses += 1
tomissfn ={}
abcessfn ={}
bopfn ={}
for ID in SubjectInfo.index: 
    missingteeth = 0
    abcesses = 0
    bleedingteeth = 0
    for tnum in toothnumbers:
       tnumdf = SiteLevelInfo.loc[(SiteLevelInfo["ID"]== ID) & (SiteLevelInfo["TOQT"] == tnum)]
       bleeddf = SiteLevelInfo.loc[(SiteLevelInfo["ID"]== ID) & (SiteLevelInfo["TOQT"] == tnum)]
       if 1 in tnumdf.TOMISSFN.values:
           missingteeth += 1
       if 1 in bleeddf.BOPFN.values:
           bleedingteeth+=1
    tomissfn.update({ID:missingteeth})
    abcessfn.update({ID:abcesses})
    bopfn.update({ID:bleedingteeth})

list = [tomissfn,abcessfn,bopfn]
collist = ["TMISSN","NABCESS","NBOP"]
x = 0
for dict in list:
    ClinValues = pd.DataFrame.from_dict(dict, orient='index',columns = [str(collist[x])])
    print(ClinValues.columns)
    FinalClinData =  pd.concat([FinalClinData,ClinValues],axis=1)
    x+=1
#AllData = pd.concat([FinalClinData,SubjectProgressionInfo],axis=)
FinalClinData['SEXCD'] = np.where(FinalClinData["SEX"]=="Male",1,2)
#AllData = FinalClinData[['ID','PDCD','SCELIGFL', 'PDCLASS', 'THERFL', 'COMPLEFL', 'SEX', 'AGE','ETHNCD', 'RACECD', 'TMISSN', 'NABCESS','NBOP']]
AllData = FinalClinData[['AGE','SEXCD','ETHNCD', 'RACECD','TMISSN', 'NBOP','CCL2_MCP_1__25_', 'IFN_gamma__29_', 'IL_6__13_', 'VEGF__26_',
       'CXCL8_IL_8__18_', 'IL_1_beta__28_','IL_10__22_',
       'Human_Osteoprotegerin__27_', 'MMP_8__27_', 'MMP_9__14_',
       'PRSSTAN','PDCD_x','PROGCLASS']]

AllData = AllData.dropna()
AllData.index


# %%
AllData = AllData.astype("float32")
AllData.columns

# %%
AllData["PROGCLASS"].values

# %%
AllData["PROGCLASS"].replace(1,0,inplace=True)
AllData["PROGCLASS"].replace(2,1,inplace=True)


# %%
AllData["PROGCLASS"].values

# %%

#only PRS logistic regression
PRSTEST = AllData[["PRSSTAN","PROGCLASS"]]
CLINTEST = AllData[['AGE','SEXCD','ETHNCD', 'RACECD','TMISSN', 'NBOP',
       'CCL2_MCP_1__25_', 'IFN_gamma__29_', 'IL_6__13_', 'VEGF__26_',
       'CXCL8_IL_8__18_', 'IL_1_beta__28_', 'MMP_8__27_', 'IL_10__22_',
       'Human_Osteoprotegerin__27_', 'MMP_9__14_','PDCD_x',"PROGCLASS"]]
from sklearn.metrics import RocCurveDisplay
from sklearn.datasets import load_wine
Q = PRSTEST.iloc[:, :-1]
r = PRSTEST.iloc[:, -1]
Q_train, Q_test, r_train, r_test = train_test_split(
    Q,r,test_size=0.2, random_state=1)

from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
def make_logreg_classifier(Q_train, Q_test, r_train, r_test, 
                        penalty = 'l2', C=1,  
                        random_state =1):
    lr = LogisticRegression(random_state = random_state, multi_class="multinomial", solver="lbfgs")
    lr.fit(Q_train, r_train)
    lr_preds = lr.predict(Q_test)
    prelr, reclr, acclr = precision_score(r_test, lr_preds,average='micro'),recall_score(r_test, lr_preds,average='micro'), lr.score(Q_test,r_test)
    
    return (prelr, reclr, acclr), lr, lr_preds

lr_scores, lr, lr_preds = make_logreg_classifier(Q_train, Q_test, r_train, r_test)
print('Logistic Regression classifier scores:')
print('Precision: {}, Recall: {}, Accuracy: {}'.format(lr_scores[0],lr_scores[1],lr_scores[2]))

lr_testing = lr

lr_vis = RocCurveDisplay.from_estimator(lr, Q_test,r_test)


#Clinical Only Testing
ClinicalFeatures = CLINTEST.iloc[:, :-1]
ClinicalTarget = CLINTEST.iloc[:, -1]
f_train, f_test, T_train, T_test = train_test_split(
    ClinicalFeatures,ClinicalTarget,test_size=0.2, random_state=1)

lr_clinscores, lrclin, lr_clinpreds = make_logreg_classifier(f_train, f_test, T_train, T_test)
print('Logistic Regression classifier scores:')
print('Precision: {}, Recall: {}, Accuracy: {}'.format(lr_clinscores[0],lr_scores[1],lr_scores[2]))

lr_clinvis = RocCurveDisplay.from_estimator(lrclin, f_test,T_test)


#PRS and Clinical logistic regression

Features = AllData.iloc[:, :-1]
Target = AllData.iloc[:, -1]
F_train, F_test, t_train, t_test = train_test_split(
    Features,Target,test_size=0.2, random_state=1)

lr_clinscores, lrclin, lr_clinpreds = make_logreg_classifier(F_train, F_test, t_train, t_test)
print('Logistic Regression classifier scores:')
print('Precision: {}, Recall: {}, Accuracy: {}'.format(lr_clinscores[0],lr_scores[1],lr_scores[2]))

lr_clinvis = RocCurveDisplay.from_estimator(lrclin, F_test,t_test)


features = AllData.iloc[:, :-1]
target = AllData.iloc[:, -1]
np.asarray(features)
np.asarray(target)
normalizer =  tf.keras.layers.Normalization(axis = -1)
normalizer.adapt(features)
X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, random_state=1)
#y_train = np.asarray(y_train).reshape(-1,1)
#y_test = np.asarray(y_test).reshape((-1,1))

input_shape = X_train.shape

print("X_train :",X_train.shape)
print("X_test:", X_test.shape)
print("X_train:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)
print("y_val:", y_test.shape)
       #normalizer,
        #tf.keras.layers.InputLayer(input_shape = X_train.shape),
nepochs = 35
def clf_model():
    clf = tf.keras.Sequential()
    opt = tf.keras.optimizers.Adam(lr=0.02)
    clf.add(tf.keras.layers.Dense(5,activation ="relu"))
    clf.add(tf.keras.layers.BatchNormalization())
    clf.add(tf.keras.layers.Dense(5,activation ="relu"))
    clf.add(tf.keras.layers.Dense(1,activation ="sigmoid")) 
    clf.compile( optimizer=opt, 
            loss = tf.keras.losses.BinaryCrossentropy(from_logits=False),
            metrics = ["accuracy",tf.keras.metrics.AUC(from_logits=False)])
    return clf

alldataclf = clf_model()
alldataclf.reset_states()
history = alldataclf.fit(X_train,y_train,epochs=nepochs,validation_split=0.15,verbose=1) # validation_split= 0.1 ,verbose=2)
plt.plot(history.history['accuracy'])
plt.title(label= " All Data Accuracy vs. Epoch")
plt.xlabel("Epoch")
plt.xticks(np.arange(0,nepochs, step=20))
plt.ylabel("% Accuracy")
alldataclf.evaluate(X_test,y_test,verbose=1)
alldataclf.count_params()
history.history


alldataclf.summary()



from sklearn.metrics import roc_curve
from sklearn.metrics import auc
y_pred_keras = alldataclf.predict(X_test).ravel()
fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_pred_keras)
auc_keras = auc(fpr_keras, tpr_keras)
plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()


shap.initjs()
explainer = shap.KernelExplainer(alldataclf.predict,X_train.iloc[:50,:])
shap_values = explainer.shap_values(X_train.iloc[50, :], nsamples=500)
shap.force_plot(explainer.expected_value) # shap_values[0], X_train.iloc[50:100,:])

shap.initjs()
shap.plots.bar(shap_values)


prsfeatures = PRSTEST.iloc[:, :-1]
prstarget = PRSTEST.iloc[:, -1]
A_train, A_test, b_train, b_test = train_test_split(
    prsfeatures, prstarget, test_size=0.2, random_state=1)

print("X_train :",A_train.shape)
print("X_test:", A_test.shape)
print("b_train:", b_train.shape)
print("b_test:", b_test.shape)

prsclf = clf_model()
prshistory = alldataclf.fit(A_train,b_train,epochs=nepochs,validation_split=0.25,verbose=1) # validation_split= 0.1 ,verbose=2)
plt.plot(prshistory.history['accuracy'])
plt.title(label= "PRS Accuracy vs. Epoch")
plt.xlabel("Epoch")
plt.xticks(np.arange(0,nepochs, step=5))
plt.ylabel("% Accuracy")
#prsclf.evaluate(A_test,b_test,verbose=1)# %%

clinfeatures = CLINTEST.iloc[:, :-1]
clintarget = CLINTEST.iloc[:, -1]
C_train, C_test, d_train, d_test = train_test_split(
    clinfeatures, clintarget.values, test_size=0.2, random_state=1)

print("Features train :",C_train.shape)
print("Features test:", C_test.shape)
print("Target train:", d_train.shape)
print("Target test:", d_test.shape)

clinclf = clf_model()
clinhistory = clinclf.fit(C_train,d_train,epochs=nepochs,validation_split=0.25,verbose=1) # validation_split= 0.1 ,verbose=2)
plt.plot(clinhistory.history['accuracy'])
plt.title(label= "Clinical Accuracy vs. Epoch")
plt.xlabel("Epoch")
plt.xticks(np.arange(0,nepochs, step=5))
plt.ylabel("% Accuracy")
clinclf.evaluate(C_test,d_test,verbose=1)
clinclf.count_params()
