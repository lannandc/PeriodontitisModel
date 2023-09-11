import tensorflow as tf
import pandas as pd
import numpy as np

SubjectInfo = pd.read_csv("/Users/lannacaruth/Documents/ads_subj_lelvel_metadata_csv.csv")
SubjectInfoInclude = SubjectInfo[SubjectInfo["COMPLEFL"]== "Y"] #remove subjects who were marked as not having completed the study
#remove dates and communication
SubjectInfoInclude.drop(["DOB","BASEDT","INFCDT","ELIGFL","ELIGDT","NOCONT","NOCNTOTH","SCRDT","MO2DT","MO4DT","MO6DT","MO8DT","MO10DT","MO12DT","PTPVDT","PTSVDT","PMO3DT","PMO6DT","COMPLEDT","DSREASCD","DSREAS","DSSPEC","LASTDT","DSCOMM"],axis=1,inplace=True)
SubjectInfoInclude.shape

SiteLevelInfo = pd.read_excel("/Users/lannacaruth/Downloads/adpd_periodontal data.xlsx") #one record per subject per visit per tooth per tooth site
SiteLevelInfoInclude = SiteLevelInfo[SiteLevelInfo["VISITN"]==0]

#select obly subject id , vist number, median pocket depth, tooth missing flags, abcess flags, tooth number, and bleeding on probing flags
toothnumbers = SiteLevelInfo["TOQT"].unique()
SiteLevelInfoIncludeClinStats = SiteLevelInfoInclude[["SUBJID","ID","VISITN","PDEP","TOMISSFN","ABCESSFN","TOQT","BOPFN"]] 
#for each id
#count values in column that are not empty
#take that number and put it in a seperate column
MouthLevelInfoIncludeClinStats = SiteLevelInfoIncludeClinStats[["SUBJID","ID"]]

SubjectProgressionInfo = pd.read_excel("/Users/lannacaruth/Downloads/Subj_level_PDClass_ProgressionClass.xlsx")
#remove index and initial classification columns
SubjectProgressionInfo.drop(["Obs","PDCLASS"],axis=1,inplace=True)
#match naming and type converntions in subject level data frame
SubjectProgressionInfo.rename(columns={"subjid":"SUBJID","progclass":"PROGCLASS"},inplace=True)
SubjectProgressionInfo["SUBJID"] = SubjectProgressionInfo["SUBJID"].astype(object)

FullSubjectLevelInfo  = pd.merge(SubjectInfoInclude,SubjectProgressionInfo,how="outer",on="SUBJID")
FullSubjectLevelInfo.nunique()

# should i drop ids with no prog class are there other files ??
FullSubjectLevelInfo["PROGCLASS"].count()
#FinalCLinData = pd.merge(FullSubjectLevelInfo,SiteLevelInfoIncludeClinStats,how="outer",on="SUBJID")

tomissfn ={}
abcessfn ={}
bopfn ={}

for ID in FullSubjectLevelInfo["SUBJID"]: 
    missingteeth = 0
    abcesses = 0
    bleedingteeth = 0
    for tnum in toothnumbers:
       tnumdf = SiteLevelInfoIncludeClinStats.loc[(SiteLevelInfoIncludeClinStats["SUBJID"]== ID) & (SiteLevelInfoIncludeClinStats["TOQT"] == tnum)]
       abcessdf = SiteLevelInfoIncludeClinStats.loc[(SiteLevelInfoIncludeClinStats["SUBJID"]== ID) & (SiteLevelInfoIncludeClinStats["TOQT"] == tnum)]
       bleeddf = SiteLevelInfoIncludeClinStats.loc[(SiteLevelInfoIncludeClinStats["SUBJID"]== ID) & (SiteLevelInfoIncludeClinStats["TOQT"] == tnum)]
       if 1 in tnumdf.TOMISSFN.values:
           missingteeth += 1
       if 1 in bleeddf.BOPFN.values:
           bleedingteeth+=1
       if 1 in abcessdf.ABCESSFN.values:
           abcesses += 1
    tomissfn.update({ID:missingteeth})
    abcessfn.update({ID:abcesses})
    bopfn.update({ID:bleedingteeth})

                                                                        
