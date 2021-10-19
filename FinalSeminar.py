from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.ensemble import BaggingRegressor
import pandas as pd
import scipy.spatial as sci
import timeit
import math
import numpy as np
from sklearn import model_selection
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import make_scorer
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor  
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.backends.backend_pdf import PdfPages
import itertools
import logging
import os
import sys
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif
from matplotlib import pyplot

# =============================================================================
class Transcript(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.logfile = open(filename, 'w')
        
    def write(self, message):
        self.terminal.write(message)
        self.logfile.write(message)
        
    def flush(self):
        pass
    
def start(filename):
    sys.stdout = Transcript(filename)
    
def stop():
    sys.stdout.logfile.close()
    sys.stdout = sys.stdout.terminal
    
# =============================================================================
def main():
    #Initiate system output
    start('initialSysOut.txt')
    #create logger
    
    logging.basicConfig(level=os.environ.get('LOGLEVEL', 'INFO'), filename="initial.txt", force = True)
    logger = logging.getLogger('FinalSeminar')
    
    # Read the original data files
    trainDF = pd.read_csv("train.csv")
    testDF = pd.read_csv("test.csv")

    #demonstrateHelpers(trainDF)
    
    #print(corrTest(trainDF))

    #trainInput, testInput, trainOutput, testIDs, predictors = transformData(trainDF, testDF)
    #trainInput, testInput, trainOutput, testIDs, predictors = beginPreprocessing(trainDF, testDF)

    
    #doExperiment(trainInput, trainOutput, predictors)
    
    #beginPreprocessing(trainDF, testDF)
    #doExperiment(trainInput, trainOutput, predictors)
    
    #hypParamTest(trainInput, trainOutput, predictors)
    
    #doKaggleTest(trainInput, testInput, trainOutput, testIDs, predictors)
    
    investigate(trainDF, testDF)

    #close system output

    stop()
    
    

    
# ===============================================================================
def readData(numRows = None):
    trainDF = pd.read_csv("data/train.csv")
    testDF = pd.read_csv("data/test.csv")
    
    outputCol = ['SalePrice']
    
    return trainDF, testDF, outputCol

def investigate(trainDF, testDF):
    
    fullDF = trainDF
    trainInput = trainDF.iloc[:, :127]
    testInput = testDF.iloc[:, :]
    
   
    
    trainOutput = trainDF.loc[:, 'Response']
    testIDs = testDF.loc[:, 'Id']
    
    '''
    
    'checking to see missing values for variables'
    
    missingPercent = trainInput.isnull().sum()/len(trainInput)
    missingPercent = missingPercent[missingPercent>0]
    
    'Make bar plot to visualize missing values'
    
    missingCols = ['Employment_Info_1','Employment_Info_4','Employment_Info_6','Insurance_History_5','Family_Hist_2','Family_Hist_3','Family_Hist_4','Family_Hist_5','Medical_History_1','Medical_History_10','Medical_History_15','Medical_History_24','Medical_History_32']
    x=range(len(missingPercent))
    fig, ax = plt.subplots()
    
    ax.barh(x, missingPercent, align='center', color='red')
    ax.set_yticks(x)
    ax.set_yticklabels(missingCols)
    ax.invert_yaxis()
    ax.set_xlabel('Percentage')
    ax.set_title('Percentage of missing values on training data')
    for i,v in enumerate(missingPercent):
        ax.text(v, i+.2, str(v), color='black', fontweight='bold')
    '''
    '''
    Since this is a classification problem, it is effected highly
    by classification in train dataset.
    We will explore classification imbalance through visualization
    '''
    #sns.set_color_codes()
    #plt.figure(figsize=(12,12))
    #sns.countplot(trainOutput, color='blue').set_title('Count of Output Class')
    
    
    'Here we are going to analyze key variables through visualizations'
    'Weight'
    '''
    f, ax = plt.subplots(1,2, figsize=(14,7))
    sns.boxplot(x='Wt', data=trainInput, orient='v', ax=ax[0])
    sns.displot(trainInput['Wt'], ax=ax[1])
    
    'Age'
    f, ax = plt.subplots(1,2, figsize=(14,7))
    sns.boxplot(x='Ins_Age', data=trainInput, orient='v', ax=ax[0])
    sns.displot(trainInput['Ins_Age'], ax=ax[1])
    
    'BMI'
    f, ax = plt.subplots(1,2, figsize=(14,7))
    sns.boxplot(x='BMI', data=trainInput, orient='v', ax=ax[0])
    sns.displot(trainInput['BMI'], ax=ax[1])
    'Height'
    f, ax = plt.subplots(1,2, figsize=(14,7))
    sns.boxplot(x='Ht', data=trainInput, orient='v', ax=ax[0])
    sns.displot(trainInput['Ht'], ax=ax[1])
    '''
    
    'Distributions and boxplots are helpful, but we could get more insight using histograms that show classification associated with variable values'
    'Lets use our histogram and pdf helper function to draw visualizations for key predictors'
    '''
    histData = fullDF
    histData = histData.rename(columns={'Response':'label'})
    
    items = {'Medical_History_1	','Medical_History_2',	'Medical_History_3',	'Medical_History_4',	'Medical_History_5',	'Medical_History_6',	'Medical_History_7',	'Medical_History_8',	'Medical_History_9',	'Medical_History_10',	'Medical_History_11', 'Medical_History_12',	'Medical_History_13',	'Medical_History_14',	'Medical_History_15',	'Medical_History_16',	'Medical_History_17',	'Medical_History_18',	'Medical_History_19',	'Medical_History_20',	'Medical_History_21',	'Medical_History_22',	'Medical_History_23',	'Medical_History_24',	'Medical_History_25',	'Medical_History_26',	'Medical_History_27',	'Medical_History_28',	'Medical_History_29',	'Medical_History_30',	'Medical_History_31',	'Medical_History_32','Medical_History_33',	'Medical_History_34',	'Medical_History_35',	'Medical_History_36',	'Medical_History_37',	'Medical_History_38',	'Medical_History_39',	'Medical_History_40',	'Medical_History_41'}
    
    pdfHist('TestHistogram.pdf', histData, items)
    
    '''
    
    'Next, we create a scatter plot to view relationship between predictors'
    
    histData = fullDF
    histData = histData.rename(columns={'Response':'label'})
    
    i = "Employment_Info_1	Employment_Info_2	Employment_Info_3	Employment_Info_4	Employment_Info_5	Employment_Info_6 InsuredInfo_1	 InsuredInfo_2	InsuredInfo_3	InsuredInfo_4	InsuredInfo_5	InsuredInfo_6	InsuredInfo_7"
    l = i.split()
    itemsList= list(itertools.permutations(l,2))
    
    #items = {'Id',	'Product_Info_1',	'Product_Info_2',	'Product_Info_3',	'Product_Info_4',	'Product_Info_5',	'Product_Info_6',	'Product_Info_7',	'Ins_Age',	'Ht',	'Wt',	'BMI'}
    #itemsList = list(itertools.permutations(items,2))
    
    pdfScatter('employmentInsuredScatter.pdf', histData, itemsList)
    
    
    
    #correlations raw data processing'
    #lets check correlation of variables
    #remove the string variable
    '''
    s = "Id	Product_Info_1	Product_Info_3	Product_Info_4	Product_Info_5	Product_Info_6	Product_Info_7	Ins_Age	Ht	Wt	BMI	Employment_Info_1	Employment_Info_2	Employment_Info_3	Employment_Info_4	Employment_Info_5	Employment_Info_6	InsuredInfo_1	InsuredInfo_2	InsuredInfo_3	InsuredInfo_4	InsuredInfo_5	InsuredInfo_6	InsuredInfo_7	Insurance_History_1	Insurance_History_2	Insurance_History_3	Insurance_History_4	Insurance_History_5	Insurance_History_7	Insurance_History_8	Insurance_History_9	Family_Hist_1	Family_Hist_2	Family_Hist_3	Family_Hist_4	Family_Hist_5	Medical_History_1	Medical_History_2	Medical_History_3	Medical_History_4	Medical_History_5	Medical_History_6	Medical_History_7	Medical_History_8	Medical_History_9	Medical_History_10	Medical_History_11	Medical_History_12	Medical_History_13	Medical_History_14	Medical_History_15	Medical_History_16	Medical_History_17	Medical_History_18	Medical_History_19	Medical_History_20	Medical_History_21	Medical_History_22	Medical_History_23	Medical_History_24	Medical_History_25	Medical_History_26	Medical_History_27	Medical_History_28	Medical_History_29	Medical_History_30	Medical_History_31	Medical_History_32	Medical_History_33	Medical_History_34	Medical_History_35	Medical_History_36	Medical_History_37	Medical_History_38	Medical_History_39	Medical_History_40	Medical_History_41	Medical_Keyword_1	Medical_Keyword_2	Medical_Keyword_3	Medical_Keyword_4	Medical_Keyword_5	Medical_Keyword_6	Medical_Keyword_7	Medical_Keyword_8	Medical_Keyword_9	Medical_Keyword_10	Medical_Keyword_11	Medical_Keyword_12	Medical_Keyword_13	Medical_Keyword_14	Medical_Keyword_15	Medical_Keyword_16	Medical_Keyword_17	Medical_Keyword_18	Medical_Keyword_19	Medical_Keyword_20	Medical_Keyword_21	Medical_Keyword_22	Medical_Keyword_23	Medical_Keyword_24	Medical_Keyword_25	Medical_Keyword_26	Medical_Keyword_27	Medical_Keyword_28	Medical_Keyword_29	Medical_Keyword_30	Medical_Keyword_31	Medical_Keyword_32	Medical_Keyword_33	Medical_Keyword_34	Medical_Keyword_35	Medical_Keyword_36	Medical_Keyword_37	Medical_Keyword_38	Medical_Keyword_39	Medical_Keyword_40	Medical_Keyword_41	Medical_Keyword_42	Medical_Keyword_43	Medical_Keyword_44	Medical_Keyword_45	Medical_Keyword_46	Medical_Keyword_47	Medical_Keyword_48"
    v = s.split()
    corrTable = corrTest(trainInput, v, trainOutput)
    y = corrTable


    #plot barplot    
    plt.figure(figsize= (25,15))
    ax = corrTable.plot(kind='bar')
    ax.set_title('Correlation with Response')
    ax.set_xlabel('Predictor')
    ax.set_ylabel('Corr')
    ax.set_xticklabels(v)
    
    plotLabels(ax)
    
    plt.savefig("corrFigureCont0")
    '''
    

    #Checking correlations after filling missing values
    #discrete variable
    '''
    trainInput.loc[:, 'Medical_History_32'].fillna(0, inplace=True)
    trainInput.loc[:, 'Medical_History_24'].fillna(0, inplace=True)
    trainInput.loc[:, 'Medical_History_15'].fillna(0, inplace=True)
    trainInput.loc[:, 'Medical_History_10'].fillna(0, inplace=True)
    trainInput.loc[:, 'Medical_History_1'].fillna(0, inplace=True)
    #continuous
    trainInput.loc[:, 'Family_Hist_5'].fillna(0, inplace=True)
    trainInput.loc[:, 'Family_Hist_4'].fillna(0, inplace=True)  
    trainInput.loc[:, 'Family_Hist_3'].fillna(0, inplace=True)
    trainInput.loc[:, 'Family_Hist_2'].fillna(0, inplace=True)    
    trainInput.loc[:, 'Employment_Info_1'].fillna(0, inplace=True)
    trainInput.loc[:, 'Employment_Info_4'].fillna(0, inplace=True)  
    trainInput.loc[:, 'Employment_Info_6'].fillna(0, inplace=True)
    trainInput.loc[:, 'Insurance_History_5'].fillna(0, inplace=True)    

    #lets check correlation of variables
    #remove the string variable
    s = "Id	Product_Info_1	Product_Info_3	Product_Info_4	Product_Info_5	Product_Info_6	Product_Info_7	Ins_Age	Ht	Wt	BMI	Employment_Info_1	Employment_Info_2	Employment_Info_3	Employment_Info_4	Employment_Info_5	Employment_Info_6	InsuredInfo_1	InsuredInfo_2	InsuredInfo_3	InsuredInfo_4	InsuredInfo_5	InsuredInfo_6	InsuredInfo_7	Insurance_History_1	Insurance_History_2	Insurance_History_3	Insurance_History_4	Insurance_History_5	Insurance_History_7	Insurance_History_8	Insurance_History_9	Family_Hist_1	Family_Hist_2	Family_Hist_3	Family_Hist_4	Family_Hist_5	Medical_History_1	Medical_History_2	Medical_History_3	Medical_History_4	Medical_History_5	Medical_History_6	Medical_History_7	Medical_History_8	Medical_History_9	Medical_History_10	Medical_History_11	Medical_History_12	Medical_History_13	Medical_History_14	Medical_History_15	Medical_History_16	Medical_History_17	Medical_History_18	Medical_History_19	Medical_History_20	Medical_History_21	Medical_History_22	Medical_History_23	Medical_History_24	Medical_History_25	Medical_History_26	Medical_History_27	Medical_History_28	Medical_History_29	Medical_History_30	Medical_History_31	Medical_History_32	Medical_History_33	Medical_History_34	Medical_History_35	Medical_History_36	Medical_History_37	Medical_History_38	Medical_History_39	Medical_History_40	Medical_History_41	Medical_Keyword_1	Medical_Keyword_2	Medical_Keyword_3	Medical_Keyword_4	Medical_Keyword_5	Medical_Keyword_6	Medical_Keyword_7	Medical_Keyword_8	Medical_Keyword_9	Medical_Keyword_10	Medical_Keyword_11	Medical_Keyword_12	Medical_Keyword_13	Medical_Keyword_14	Medical_Keyword_15	Medical_Keyword_16	Medical_Keyword_17	Medical_Keyword_18	Medical_Keyword_19	Medical_Keyword_20	Medical_Keyword_21	Medical_Keyword_22	Medical_Keyword_23	Medical_Keyword_24	Medical_Keyword_25	Medical_Keyword_26	Medical_Keyword_27	Medical_Keyword_28	Medical_Keyword_29	Medical_Keyword_30	Medical_Keyword_31	Medical_Keyword_32	Medical_Keyword_33	Medical_Keyword_34	Medical_Keyword_35	Medical_Keyword_36	Medical_Keyword_37	Medical_Keyword_38	Medical_Keyword_39	Medical_Keyword_40	Medical_Keyword_41	Medical_Keyword_42	Medical_Keyword_43	Medical_Keyword_44	Medical_Keyword_45	Medical_Keyword_46	Medical_Keyword_47	Medical_Keyword_48"
    v = s.split()
    corrTable = corrTest(trainInput, v, trainOutput)
    y = corrTable


    #plot barplot    
    plt.figure(figsize= (25,15))
    ax = corrTable.plot(kind='bar')
    ax.set_title('Correlation with Response')
    ax.set_xlabel('Predictor')
    ax.set_ylabel('Corr')
    ax.set_xticklabels(v)
    
    plotLabels(ax)
    
    plt.savefig("corrFigureCont0")
    '''
    
    #heatmaps to check dummy variables
    '''
    cols = 'Medical_Keyword_1	Medical_Keyword_2	Medical_Keyword_3	Medical_Keyword_4	Medical_Keyword_5	Medical_Keyword_6	Medical_Keyword_7	Medical_Keyword_8	Medical_Keyword_9	Medical_Keyword_10	Medical_Keyword_11	Medical_Keyword_12	Medical_Keyword_13	Medical_Keyword_14	Medical_Keyword_15	Medical_Keyword_16	Medical_Keyword_17	Medical_Keyword_18	Medical_Keyword_19	Medical_Keyword_20	Medical_Keyword_21	Medical_Keyword_22	Medical_Keyword_23	Medical_Keyword_24	Medical_Keyword_25	Medical_Keyword_26	Medical_Keyword_27	Medical_Keyword_28	Medical_Keyword_29	Medical_Keyword_30	Medical_Keyword_31	Medical_Keyword_32	Medical_Keyword_33	Medical_Keyword_34	Medical_Keyword_35	Medical_Keyword_36	Medical_Keyword_37	Medical_Keyword_38	Medical_Keyword_39	Medical_Keyword_40	Medical_Keyword_41	Medical_Keyword_42	Medical_Keyword_43	Medical_Keyword_44	Medical_Keyword_45	Medical_Keyword_46	Medical_Keyword_47	Medical_Keyword_48'
    colList= cols.split()
    correlation = trainInput.loc[:,colList].corr()
    plt.figure(figsize=(50,50))
    sns.heatmap(correlation, vmax=1, square=True,annot=True,cmap='cubehelix')
    plt.title('Correlation between Medical_Keywords')
    '''



def beginPreprocessing (trainDF, testDF):
    
    'divide the data frames into testing and training sets'
    fullDF = trainDF
    trainInput = trainDF.iloc[:, :127]
    testInput = testDF.iloc[:, :]
    
    trainOutput = trainDF.loc[:, 'Response']
    testIDs = testDF.loc[:, 'Id']
    
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 200)

    
    'making predictors'
    #s = "Id	Product_Info_1 	Product_Info_3	Product_Info_4	Product_Info_5	Product_Info_6	Product_Info_7	Ins_Age	Ht	Wt	BMI	Employment_Info_1	Employment_Info_2	Employment_Info_3	Employment_Info_4	Employment_Info_5	Employment_Info_6	InsuredInfo_1	InsuredInfo_2	InsuredInfo_3	InsuredInfo_4	InsuredInfo_5	InsuredInfo_6	InsuredInfo_7	Insurance_History_1	Insurance_History_2	Insurance_History_3	Insurance_History_4	Insurance_History_5	Insurance_History_7	Insurance_History_8	Insurance_History_9	Family_Hist_1	Family_Hist_2	Family_Hist_3	Family_Hist_4	Family_Hist_5	Medical_History_1	Medical_History_2	Medical_History_3	Medical_History_4	Medical_History_5	Medical_History_6	Medical_History_7	Medical_History_8	Medical_History_9	Medical_History_10	Medical_History_11	Medical_History_12	Medical_History_13	Medical_History_14	Medical_History_15	Medical_History_16	Medical_History_17	Medical_History_18	Medical_History_19	Medical_History_20	Medical_History_21	Medical_History_22	Medical_History_23	Medical_History_24	Medical_History_25	Medical_History_26	Medical_History_27	Medical_History_28	Medical_History_29	Medical_History_30	Medical_History_31	Medical_History_32	Medical_History_33	Medical_History_34	Medical_History_35	Medical_History_36	Medical_History_37	Medical_History_38	Medical_History_39	Medical_History_40	Medical_History_41	Medical_Keyword_1	Medical_Keyword_2	Medical_Keyword_3	Medical_Keyword_4	Medical_Keyword_5	Medical_Keyword_6	Medical_Keyword_7	Medical_Keyword_8	Medical_Keyword_9	Medical_Keyword_10	Medical_Keyword_11	Medical_Keyword_12	Medical_Keyword_13	Medical_Keyword_14	Medical_Keyword_15	Medical_Keyword_16	Medical_Keyword_17	Medical_Keyword_18	Medical_Keyword_19	Medical_Keyword_20	Medical_Keyword_21	Medical_Keyword_22	Medical_Keyword_23	Medical_Keyword_24	Medical_Keyword_25	Medical_Keyword_26	Medical_Keyword_27	Medical_Keyword_28	Medical_Keyword_29	Medical_Keyword_30	Medical_Keyword_31	Medical_Keyword_32	Medical_Keyword_33	Medical_Keyword_34	Medical_Keyword_35	Medical_Keyword_36	Medical_Keyword_37	Medical_Keyword_38	Medical_Keyword_39	Medical_Keyword_40	Medical_Keyword_41	Medical_Keyword_42	Medical_Keyword_43	Medical_Keyword_44	Medical_Keyword_45	Medical_Keyword_46	Medical_Keyword_47	Medical_Keyword_48"
    s = "Product_Info_1 Product_Info_2 	Product_Info_3	Product_Info_4	Product_Info_5	Product_Info_6	Product_Info_7	Ins_Age	Ht	Wt	BMI	Employment_Info_1	Employment_Info_2	Employment_Info_3	Employment_Info_4	Employment_Info_5	Employment_Info_6	InsuredInfo_1	InsuredInfo_2	InsuredInfo_3	InsuredInfo_4	InsuredInfo_5	InsuredInfo_6	InsuredInfo_7	Insurance_History_1	Insurance_History_2	Insurance_History_3	Insurance_History_4	Insurance_History_5	Insurance_History_7	Insurance_History_8	Insurance_History_9	Family_Hist_1	Family_Hist_2	Family_Hist_3	Family_Hist_4	Family_Hist_5	Medical_History_1	Medical_History_2	Medical_History_3	Medical_History_4	Medical_History_5	Medical_History_6	Medical_History_7	Medical_History_8	Medical_History_9	Medical_History_10	Medical_History_11	Medical_History_12	Medical_History_13	Medical_History_14	Medical_History_15	Medical_History_16	Medical_History_17	Medical_History_18	Medical_History_19	Medical_History_20	Medical_History_21	Medical_History_22	Medical_History_23	Medical_History_24	Medical_History_25	Medical_History_26	Medical_History_27	Medical_History_28	Medical_History_29	Medical_History_30	Medical_History_31	Medical_History_32	Medical_History_33	Medical_History_34	Medical_History_35	Medical_History_36	Medical_History_37	Medical_History_38	Medical_History_39	Medical_History_40	Medical_History_41	Medical_Keyword_1	Medical_Keyword_2	Medical_Keyword_3	Medical_Keyword_4	Medical_Keyword_5	Medical_Keyword_6	Medical_Keyword_7	Medical_Keyword_8	Medical_Keyword_9	Medical_Keyword_10	Medical_Keyword_11	Medical_Keyword_12	Medical_Keyword_13	Medical_Keyword_14	Medical_Keyword_15	Medical_Keyword_16	Medical_Keyword_17	Medical_Keyword_18	Medical_Keyword_19	Medical_Keyword_20	Medical_Keyword_21	Medical_Keyword_22	Medical_Keyword_23	Medical_Keyword_24	Medical_Keyword_25	Medical_Keyword_26	Medical_Keyword_27	Medical_Keyword_28	Medical_Keyword_29	Medical_Keyword_30	Medical_Keyword_31	Medical_Keyword_32	Medical_Keyword_33	Medical_Keyword_34	Medical_Keyword_35	Medical_Keyword_36	Medical_Keyword_37	Medical_Keyword_38	Medical_Keyword_39	Medical_Keyword_40	Medical_Keyword_41	Medical_Keyword_42	Medical_Keyword_43	Medical_Keyword_44	Medical_Keyword_45	Medical_Keyword_46	Medical_Keyword_47	Medical_Keyword_48 productInfoLetter	productInfoInt	medKeyCount	binaryMean	binaryMin	binaryMax"

    #Lets drop some variables for purpose of test right now
    #s = "Id	Product_Info_1	Product_Info_3	Product_Info_4	Product_Info_5	Product_Info_6	Product_Info_7	Ins_Age	Ht	Wt	BMI		Employment_Info_2	Employment_Info_3	Employment_Info_4	Employment_Info_5	Employment_Info_6	InsuredInfo_1	InsuredInfo_2	InsuredInfo_3	InsuredInfo_4	InsuredInfo_5	InsuredInfo_6	InsuredInfo_7	Insurance_History_1	Insurance_History_2	Insurance_History_3	Insurance_History_4	Insurance_History_5	Insurance_History_7	Insurance_History_8	Insurance_History_9	Family_Hist_1	Family_Hist_2	Family_Hist_3				Medical_History_2	Medical_History_3	Medical_History_4	Medical_History_5	Medical_History_6	Medical_History_7	Medical_History_8	Medical_History_9		Medical_History_11	Medical_History_12	Medical_History_13	Medical_History_14		Medical_History_16	Medical_History_17	Medical_History_18	Medical_History_19	Medical_History_20	Medical_History_21	Medical_History_22	Medical_History_23		Medical_History_25	Medical_History_26	Medical_History_27	Medical_History_28	Medical_History_29	Medical_History_30	Medical_History_31		Medical_History_33	Medical_History_34	Medical_History_35	Medical_History_36	Medical_History_37	Medical_History_38	Medical_History_39	Medical_History_40	Medical_History_41	Medical_Keyword_1	Medical_Keyword_2	Medical_Keyword_3	Medical_Keyword_4	Medical_Keyword_5	Medical_Keyword_6	Medical_Keyword_7	Medical_Keyword_8	Medical_Keyword_9	Medical_Keyword_10	Medical_Keyword_11	Medical_Keyword_12	Medical_Keyword_13	Medical_Keyword_14	Medical_Keyword_15	Medical_Keyword_16	Medical_Keyword_17	Medical_Keyword_18	Medical_Keyword_19	Medical_Keyword_20	Medical_Keyword_21	Medical_Keyword_22	Medical_Keyword_23	Medical_Keyword_24	Medical_Keyword_25	Medical_Keyword_26	Medical_Keyword_27	Medical_Keyword_28	Medical_Keyword_29	Medical_Keyword_30	Medical_Keyword_31	Medical_Keyword_32	Medical_Keyword_33	Medical_Keyword_34	Medical_Keyword_35	Medical_Keyword_36	Medical_Keyword_37	Medical_Keyword_38	Medical_Keyword_39	Medical_Keyword_40	Medical_Keyword_41	Medical_Keyword_42	Medical_Keyword_43	Medical_Keyword_44	Medical_Keyword_45	Medical_Keyword_46	Medical_Keyword_47	Medical_Keyword_48"

    v = s.split()

    #drop product Info 2 for purpose of tests
    predictors = v
    

        
    'missing values'
    #Using our missing value plot from the investigate method, I identified missing values.
    #I use correlation with the response variable to determine whether filling missing values
    #in a particular way is efficient or not. I first calculate correlation before any processing, and then after fillna()
    #Since all of the missing predictors are either continous or discrete, I can replace them easily. 
    #My correlation tests show that predictors with a >20% missing values do well when 
    #the missing values are replaced with 0. For lower percentage missed values, I use the mean.
    
    #<20% missing value predictors:
    trainInput.loc[:, 'Employment_Info_1'].fillna(trainInput.loc[:, 'Employment_Info_1'].mean(), inplace=True)
    trainInput.loc[:, 'Employment_Info_4'].fillna(trainInput.loc[:, 'Employment_Info_4'].mean(), inplace=True) 
    trainInput.loc[:, 'Employment_Info_6'].fillna(trainInput.loc[:, 'Employment_Info_6'].mean(), inplace=True)  
    trainInput.loc[:, 'Medical_History_1'].fillna(trainInput.loc[:, 'Medical_History_1'].mean(), inplace=True)  
    #>20% missing value predictors:
    trainInput.loc[:, 'Medical_History_32'].fillna(0, inplace=True)
    trainInput.loc[:, 'Medical_History_24'].fillna(0, inplace=True)
    trainInput.loc[:, 'Medical_History_15'].fillna(0, inplace=True)
    trainInput.loc[:, 'Medical_History_10'].fillna(0, inplace=True)    
    trainInput.loc[:, 'Family_Hist_5'].fillna(0, inplace=True)
    trainInput.loc[:, 'Family_Hist_4'].fillna(0, inplace=True)  
    trainInput.loc[:, 'Family_Hist_3'].fillna(0, inplace=True)
    trainInput.loc[:, 'Family_Hist_2'].fillna(0, inplace=True)        
    trainInput.loc[:, 'Insurance_History_5'].fillna(0, inplace=True)   
    
    #testInput:
    #<20% missing value predictors:
    testInput.loc[:, 'Employment_Info_1'].fillna(testInput.loc[:, 'Employment_Info_1'].mean(), inplace=True)
    testInput.loc[:, 'Employment_Info_4'].fillna(testInput.loc[:, 'Employment_Info_4'].mean(), inplace=True) 
    testInput.loc[:, 'Employment_Info_6'].fillna(testInput.loc[:, 'Employment_Info_6'].mean(), inplace=True)  
    testInput.loc[:, 'Medical_History_1'].fillna(testInput.loc[:, 'Medical_History_1'].mean(), inplace=True)  
    #>20% missing value predictors:
    testInput.loc[:, 'Medical_History_32'].fillna(0, inplace=True)
    testInput.loc[:, 'Medical_History_24'].fillna(0, inplace=True)
    testInput.loc[:, 'Medical_History_15'].fillna(0, inplace=True)
    testInput.loc[:, 'Medical_History_10'].fillna(0, inplace=True)    
    testInput.loc[:, 'Family_Hist_5'].fillna(0, inplace=True)
    testInput.loc[:, 'Family_Hist_4'].fillna(0, inplace=True)  
    testInput.loc[:, 'Family_Hist_3'].fillna(0, inplace=True)
    testInput.loc[:, 'Family_Hist_2'].fillna(0, inplace=True)        
    testInput.loc[:, 'Insurance_History_5'].fillna(0, inplace=True)           


    'normalization is not needed since data is already normal'
    
    'BMI-WT'
    #Looking at the scaterplots from investigate, it is apparent that BMI and weight
    #are significant determinators of the response variable. 
    #Thus, I will make a couple of dummy variables to account for this.
    
    #create new predictor = Wt*BMI
    trainDF['Wt*BMI'] = trainDF['Wt']*trainDF['BMI']
    
    
    #The scatter plot and further analysis in Excel tells us that BMI >0.85 and Wt>0.4 
    #are predominately classified as high risk (level1, level2). Lets add the dummy variable
    trainDF['bmiHtCond'] = np.where((trainDF['BMI']>0.85) & (trainDF['Wt']>0.4), 1, 0)
    
    
    #'trainDF.to_csv('responseCheck/bmiHtCond.csv', index=False)'
    
    
    'BMI-Product_Info_7'    
    #The scatter plot and further analysis in Excel tells us that BMI >0.85 and Wt>0.4 
    #are predominately classified as high risk (level1, level2). Lets add the dummy variable    
    trainDF['bmiProd7'] = np.where((trainDF['BMI']>0.85) & (trainDF['Product_Info_7']==1), 1, 0)
    
    
    'Ins_Age'
    #According to hist plot, proportion of high risk classification drastically increases on Ins_Age>0.7
    trainDF['Agehigh'] = np.where((trainDF['Ins_Age']>0.7), 1, 0)
    
    
    'Product_Info_2'
    #Product Info 2  variable is comprised of a character and number, we can seperate these
    #to make two new variabels, and then apply label encoding. 
    
    trainInput['productInfoLetter'] = trainInput.Product_Info_2.str[0]
    trainInput['productInfoInt'] = trainInput.Product_Info_2.str[1]
    testInput['productInfoLetter'] = testInput.Product_Info_2.str[0]
    testInput['productInfoInt'] = testInput.Product_Info_2.str[1]
    
    trainInput['productInfoLetter'] = LabelEncoder().fit_transform(trainInput.productInfoLetter)
    trainInput['productInfoInt'] = LabelEncoder().fit_transform(trainInput.productInfoInt)
    trainInput['Product_Info_2'] = LabelEncoder().fit_transform(trainInput.Product_Info_2)
    testInput['productInfoLetter'] = LabelEncoder().fit_transform(testInput.productInfoLetter)
    testInput['productInfoInt'] = LabelEncoder().fit_transform(testInput.productInfoInt)
    testInput['Product_Info_2'] = LabelEncoder().fit_transform(testInput.Product_Info_2)
    

    'MedicalKeywords'
    #The medical keywords are all dummy variables. The correlation heatmap in investigate
    #shows that there is no correlation between these words. Consequently, Im making a 
    #new predictor that sums the medical keywords.
    
    medKeyCols = 'Medical_Keyword_1	Medical_Keyword_2	Medical_Keyword_3	Medical_Keyword_4	Medical_Keyword_5	Medical_Keyword_6	Medical_Keyword_7	Medical_Keyword_8	Medical_Keyword_9	Medical_Keyword_10	Medical_Keyword_11	Medical_Keyword_12	Medical_Keyword_13	Medical_Keyword_14	Medical_Keyword_15	Medical_Keyword_16	Medical_Keyword_17	Medical_Keyword_18	Medical_Keyword_19	Medical_Keyword_20	Medical_Keyword_21	Medical_Keyword_22	Medical_Keyword_23	Medical_Keyword_24	Medical_Keyword_25	Medical_Keyword_26	Medical_Keyword_27	Medical_Keyword_28	Medical_Keyword_29	Medical_Keyword_30	Medical_Keyword_31	Medical_Keyword_32	Medical_Keyword_33	Medical_Keyword_34	Medical_Keyword_35	Medical_Keyword_36	Medical_Keyword_37	Medical_Keyword_38	Medical_Keyword_39	Medical_Keyword_40	Medical_Keyword_41	Medical_Keyword_42	Medical_Keyword_43	Medical_Keyword_44	Medical_Keyword_45	Medical_Keyword_46	Medical_Keyword_47	Medical_Keyword_48'
    medKeyColsList= medKeyCols.split()
    trainInput['medKeyCount'] = trainInput.loc[:, medKeyColsList].sum(axis=1)
    testInput['medKeyCount'] = testInput.loc[:, medKeyColsList].sum(axis=1)
    
    
    'Mean target encoding'
    #features have a non-linear target dependency, so let us apply mean encoding!
    #to avoid overfitting, the data will be encoded using k-folds.
    
    trainInput['Product_Info_1'], testInput['Product_Info_1'] = targetEncoding(trainDF, testInput, ['Product_Info_1'], 'Response')
    trainInput['Product_Info_2'], testInput['Product_Info_2'] = targetEncoding(trainDF, testInput, ['Product_Info_2'], 'Response')
    trainInput['Product_Info_3'], testInput['Product_Info_3'] = targetEncoding(trainDF, testInput, ['Product_Info_3'], 'Response')
    trainInput['Product_Info_5'], testInput['Product_Info_5'] = targetEncoding(trainDF, testInput, ['Product_Info_5'], 'Response')
    trainInput['Product_Info_6'], testInput['Product_Info_6'] = targetEncoding(trainDF, testInput, ['Product_Info_6'], 'Response')
    trainInput['Product_Info_7'], testInput['Product_Info_7'] = targetEncoding(trainDF, testInput, ['Product_Info_7'], 'Response')   
    trainInput['Employment_Info_2'], testInput['Employment_Info_2'] = targetEncoding(trainDF, testInput, ['Employment_Info_2'], 'Response')      
    trainInput['Employment_Info_3'], testInput['Employment_Info_3'] = targetEncoding(trainDF, testInput, ['Employment_Info_3'], 'Response')          
    trainInput['Employment_Info_5'], testInput['Employment_Info_5'] = targetEncoding(trainDF, testInput, ['Employment_Info_5'], 'Response')          
    trainInput['InsuredInfo_1'], testInput['InsuredInfo_1'] = targetEncoding(trainDF, testInput, ['InsuredInfo_1'], 'Response')       
    trainInput['InsuredInfo_2'], testInput['InsuredInfo_2'] = targetEncoding(trainDF, testInput, ['InsuredInfo_2'], 'Response')           
    trainInput['InsuredInfo_3'], testInput['InsuredInfo_3'] = targetEncoding(trainDF, testInput, ['InsuredInfo_3'], 'Response')       
    trainInput['InsuredInfo_4'], testInput['InsuredInfo_4'] = targetEncoding(trainDF, testInput, ['InsuredInfo_4'], 'Response')      
    trainInput['InsuredInfo_5'], testInput['InsuredInfo_5'] = targetEncoding(trainDF, testInput, ['InsuredInfo_5'], 'Response')       
    trainInput['InsuredInfo_6'], testInput['InsuredInfo_6'] = targetEncoding(trainDF, testInput, ['InsuredInfo_6'], 'Response')      
    trainInput['InsuredInfo_7'], testInput['InsuredInfo_7'] = targetEncoding(trainDF, testInput, ['InsuredInfo_7'], 'Response')       
    trainInput['Insurance_History_1'], testInput['Insurance_History_1'] = targetEncoding(trainDF, testInput, ['Insurance_History_1'], 'Response')      
    trainInput['Insurance_History_2'], testInput['Insurance_History_2'] = targetEncoding(trainDF, testInput, ['Insurance_History_2'], 'Response')      
    trainInput['Insurance_History_3'], testInput['Insurance_History_3'] = targetEncoding(trainDF, testInput, ['Insurance_History_3'], 'Response')      
    trainInput['Insurance_History_4'], testInput['Insurance_History_4'] = targetEncoding(trainDF, testInput, ['Insurance_History_4'], 'Response')      
    trainInput['Insurance_History_7'], testInput['Insurance_History_7'] = targetEncoding(trainDF, testInput, ['Insurance_History_7'], 'Response')      
    trainInput['Insurance_History_8'], testInput['Insurance_History_8'] = targetEncoding(trainDF, testInput, ['Insurance_History_8'], 'Response')      
    trainInput['Insurance_History_9'], testInput['Insurance_History_9'] = targetEncoding(trainDF, testInput, ['Insurance_History_9'], 'Response')      
    trainInput['Family_Hist_1'], testInput['Family_Hist_1'] = targetEncoding(trainDF, testInput, ['Family_Hist_1'], 'Response')      
    trainInput['Medical_History_2'], testInput['Medical_History_2'] = targetEncoding(trainDF, testInput, ['Medical_History_2'], 'Response')      
    trainInput['Medical_History_3'], testInput['Medical_History_3'] = targetEncoding(trainDF, testInput, ['Medical_History_3'], 'Response')      
    trainInput['Medical_History_4'], testInput['Medical_History_4'] = targetEncoding(trainDF, testInput, ['Medical_History_4'], 'Response')      
    trainInput['Medical_History_5'], testInput['Medical_History_5'] = targetEncoding(trainDF, testInput, ['Medical_History_5'], 'Response')      
    trainInput['Medical_History_6'], testInput['Medical_History_6'] = targetEncoding(trainDF, testInput, ['Medical_History_6'], 'Response')      
    trainInput['Medical_History_7'], testInput['Medical_History_7'] = targetEncoding(trainDF, testInput, ['Medical_History_7'], 'Response')      
    trainInput['Medical_History_8'], testInput['Medical_History_8'] = targetEncoding(trainDF, testInput, ['Medical_History_8'], 'Response')      
    trainInput['Medical_History_9'], testInput['Medical_History_9'] = targetEncoding(trainDF, testInput, ['Medical_History_9'], 'Response')      
    trainInput['Medical_History_11'], testInput['Medical_History_11'] = targetEncoding(trainDF, testInput, ['Medical_History_11'], 'Response')      
    trainInput['Medical_History_12'], testInput['Medical_History_12'] = targetEncoding(trainDF, testInput, ['Medical_History_12'], 'Response')      
    trainInput['Medical_History_13'], testInput['Medical_History_13'] = targetEncoding(trainDF, testInput, ['Medical_History_13'], 'Response')      
    trainInput['Medical_History_14'], testInput['Medical_History_14'] = targetEncoding(trainDF, testInput, ['Medical_History_14'], 'Response')      
    trainInput['Medical_History_16'], testInput['Medical_History_16'] = targetEncoding(trainDF, testInput, ['Medical_History_16'], 'Response')      
    trainInput['Medical_History_17'], testInput['Medical_History_17'] = targetEncoding(trainDF, testInput, ['Medical_History_17'], 'Response')      
    trainInput['Medical_History_18'], testInput['Medical_History_18'] = targetEncoding(trainDF, testInput, ['Medical_History_18'], 'Response')      
    trainInput['Medical_History_19'], testInput['Medical_History_19'] = targetEncoding(trainDF, testInput, ['Medical_History_19'], 'Response')      
    trainInput['Medical_History_20'], testInput['Medical_History_20'] = targetEncoding(trainDF, testInput, ['Medical_History_20'], 'Response')      
    trainInput['Medical_History_21'], testInput['Medical_History_21'] = targetEncoding(trainDF, testInput, ['Medical_History_21'], 'Response')      
    trainInput['Medical_History_22'], testInput['Medical_History_22'] = targetEncoding(trainDF, testInput, ['Medical_History_22'], 'Response')      
    trainInput['Medical_History_23'], testInput['Medical_History_23'] = targetEncoding(trainDF, testInput, ['Medical_History_23'], 'Response')      
    trainInput['Medical_History_25'], testInput['Medical_History_25'] = targetEncoding(trainDF, testInput, ['Medical_History_25'], 'Response')      
    trainInput['Medical_History_26'], testInput['Medical_History_26'] = targetEncoding(trainDF, testInput, ['Medical_History_26'], 'Response')      
    trainInput['Medical_History_27'], testInput['Medical_History_27'] = targetEncoding(trainDF, testInput, ['Medical_History_27'], 'Response')      
    trainInput['Medical_History_28'], testInput['Medical_History_28'] = targetEncoding(trainDF, testInput, ['Medical_History_28'], 'Response')      
    trainInput['Medical_History_29'], testInput['Medical_History_29'] = targetEncoding(trainDF, testInput, ['Medical_History_29'], 'Response')      
    trainInput['Medical_History_30'], testInput['Medical_History_30'] = targetEncoding(trainDF, testInput, ['Medical_History_30'], 'Response')      
    trainInput['Medical_History_31'], testInput['Medical_History_31'] = targetEncoding(trainDF, testInput, ['Medical_History_31'], 'Response')      
    trainInput['Medical_History_33'], testInput['Medical_History_33'] = targetEncoding(trainDF, testInput, ['Medical_History_33'], 'Response')      
    trainInput['Medical_History_34'], testInput['Medical_History_34'] = targetEncoding(trainDF, testInput, ['Medical_History_34'], 'Response')      
    trainInput['Medical_History_35'], testInput['Medical_History_35'] = targetEncoding(trainDF, testInput, ['Medical_History_35'], 'Response')      
    trainInput['Medical_History_36'], testInput['Medical_History_36'] = targetEncoding(trainDF, testInput, ['Medical_History_36'], 'Response')      
    trainInput['Medical_History_37'], testInput['Medical_History_37'] = targetEncoding(trainDF, testInput, ['Medical_History_37'], 'Response')      
    trainInput['Medical_History_38'], testInput['Medical_History_38'] = targetEncoding(trainDF, testInput, ['Medical_History_38'], 'Response')      
    trainInput['Medical_History_39'], testInput['Medical_History_39'] = targetEncoding(trainDF, testInput, ['Medical_History_39'], 'Response')      
    trainInput['Medical_History_40'], testInput['Medical_History_40'] = targetEncoding(trainDF, testInput, ['Medical_History_40'], 'Response')      
    trainInput['Medical_History_41'], testInput['Medical_History_41'] = targetEncoding(trainDF, testInput, ['Medical_History_41'], 'Response')      
   
    'Drop ID Variable'
    testInput.drop(['Id'], axis=1, inplace=True)
    trainInput.drop(['Id'], axis =1, inplace=True)
    
    
    'binaryVariables'
    #There are 48 binary variables in the dataset, lets apply a version of
    #target mean encoding to these variables
    
    s = "Medical_Keyword_1	Medical_Keyword_2	Medical_Keyword_3	Medical_Keyword_4	Medical_Keyword_5	Medical_Keyword_6	Medical_Keyword_7	Medical_Keyword_8	Medical_Keyword_9	Medical_Keyword_10	Medical_Keyword_11	Medical_Keyword_12	Medical_Keyword_13	Medical_Keyword_14	Medical_Keyword_15	Medical_Keyword_16	Medical_Keyword_17	Medical_Keyword_18	Medical_Keyword_19	Medical_Keyword_20	Medical_Keyword_21	Medical_Keyword_22	Medical_Keyword_23	Medical_Keyword_24	Medical_Keyword_25	Medical_Keyword_26	Medical_Keyword_27	Medical_Keyword_28	Medical_Keyword_29	Medical_Keyword_30	Medical_Keyword_31	Medical_Keyword_32	Medical_Keyword_33	Medical_Keyword_34	Medical_Keyword_35	Medical_Keyword_36	Medical_Keyword_37	Medical_Keyword_38	Medical_Keyword_39	Medical_Keyword_40	Medical_Keyword_41	Medical_Keyword_42	Medical_Keyword_43	Medical_Keyword_44	Medical_Keyword_45	Medical_Keyword_46	Medical_Keyword_47	Medical_Keyword_48"
    v =s.split()
    
    binaryTrain, binaryTest = binaryManipulation(trainDF, testInput, v, 'Response')
    #trainInput = pd.concat([trainInput.reset_index(drop=True), binaryTrain.reset_index(drop=True)], sort=False)
    #testInput = pd.concat([testInput.reset_index(drop=True), binaryTest.reset_index(drop=True)], sort=False)
    trainInput = pd.concat([trainInput, binaryTrain], axis=1, sort=False)
    testInput = pd.concat([testInput, binaryTest], axis=1, sort=False)
    



    
    'feature selection / dimensionality reduction'
    #featureSelectionCat(trainInput, trainOutput, testInput)
    #featureSelectionMIF(trainInput, trainOutput, testInput)
    
    #dropping variables
    testInput.drop(['Product_Info_6', 'Product_Info_7', 'InsuredInfo_4', 'Insurance_History_1', 'Insurance_History_3', 'Insurance_History_8', 'Medical_History_3', 'Medical_History_11','Medical_History_12', 'Medical_History_14', 'Medical_History_25', 'Medical_History_35', 'Medical_Keyword_5', 'Medical_Keyword_6','Medical_Keyword_7', 'Medical_Keyword_28','Medical_Keyword_29'], axis=1, inplace=True)
    trainInput.drop(['Product_Info_6', 'Product_Info_7', 'InsuredInfo_4', 'Insurance_History_1', 'Insurance_History_3', 'Insurance_History_8', 'Medical_History_3', 'Medical_History_11','Medical_History_12', 'Medical_History_14', 'Medical_History_25', 'Medical_History_35', 'Medical_Keyword_5', 'Medical_Keyword_6','Medical_Keyword_7', 'Medical_Keyword_28','Medical_Keyword_29'], axis =1, inplace=True)
    
    
    print('preprocessingComplete')
    
    'return cleaned data'    
    return trainInput, testInput, trainOutput, testIDs, predictors
    
    
    
    
def corrTest(df, colNames, test):
    
    corrDF = df.loc[:, colNames]
    
    corrFigures = corrDF.apply(lambda col: col.corr(test) ,axis =0)
    
    return corrFigures

#Chi2 classification feature selection method
def featureSelectionCat(trainInput, trainOutput, testInput):
    
     alg = SelectKBest(score_func=chi2, k='all')
     alg.fit(trainInput, trainOutput)
     trainedInput = alg.transform(trainInput)
     testedInput = alg.transform(testInput)
     
     #printoutputs
     for i in range(len(alg.scores_)):
         print('Predictor %d: %f' % (i, alg.scores_[i]))
        
     pyplot.bar([i for i in range(len(alg.scores_))], alg.scores_)
     pyplot.show()
         
#Mutual Information Feature Selection Method
def featureSelectionMIF(trainInput, trainOutput, testInput):
    
     alg = SelectKBest(score_func=mutual_info_classif, k='all')
     alg.fit(trainInput, trainOutput)
     trainedInput = alg.transform(trainInput)
     testedInput = alg.transform(testInput)
     
     #printoutputs
     for i in range(len(alg.scores_)):
         print('Predictor %d: %f' % (i, alg.scores_[i]))
    
     pyplot.bar([i for i in range(len(alg.scores_))], alg.scores_)
     pyplot.show()


def targetEncoding(trainInput, testInput, cols, target, n_folds = 10):
    trainCopy , testCopy = trainInput.copy(), testInput.copy()
    kf = KFold(n_splits = n_folds, random_state=(1500), shuffle=True)
    
    for col in cols:
        #calculate a mean value for categories that equal zero due to their low number
        missingMean = trainCopy[target].mean()
        #make dictionary to keep track of fold results
        Dictionary = {}
        i=1
        for trIn, testIn in kf.split(trainInput):
            targetMean = trainCopy.iloc[trIn].groupby(col)[target].mean()
            Dictionary['fold'+ str(i)]=targetMean
            i+=1
            trainCopy.loc[testIn, col+'Enc'] = trainCopy.loc[testIn, col].map(targetMean)
        trainCopy[col+'Enc'].fillna(missingMean, inplace=True)
        #test set + dictionary
        fold1 = Dictionary.get('fold1')
        fold2 = Dictionary.get('fold2')
        fold3 = Dictionary.get('fold3')
        fold4 = Dictionary.get('fold4')
        fold5 = Dictionary.get('fold5')
        fold6 = Dictionary.get('fold6')
        fold7 = Dictionary.get('fold7')
        fold8 = Dictionary.get('fold8')
        fold9 = Dictionary.get('fold9')
        fold10 = Dictionary.get('fold10')       
        
        #applying mean of all folds to test dataset
        folds = pd.concat([fold1, fold2, fold3, fold4, fold5, fold6,fold7, fold8, fold9, fold10], axis=1)

        foldsMean = folds.mean(axis=1)                
        #colMean = trainCopy.groupby(col)[target].mean()
        testCopy[col+'Enc'] = testCopy[col].map(foldsMean)
        testCopy[col+'Enc'].fillna(missingMean, inplace=True)
            
    trainCopy = trainCopy.filter(like = 'Enc', axis=1)
    testCopy = testCopy.filter(like = 'Enc', axis=1)
    return trainCopy, testCopy
    

def binaryManipulation(trainInput, testInput, cols, target, n_folds = 10):
    trainCopy , testCopy = trainInput.copy(), testInput.copy()
    kf = KFold(n_splits = n_folds, random_state=(1500), shuffle=True)
    
    for col in cols:

        #make dictionary to keep track of fold results
        Dictionary = {}
        i=0
        for trIn, testIn in kf.split(trainInput):
            targetMean = trainCopy.iloc[trIn].groupby(col)[target].mean()
            Dictionary[str(col)+'enc'+str(i)]=targetMean
            i+=1
            trainCopy.loc[testIn, col+'Enc'] = trainCopy.loc[testIn, col].map(targetMean)
        #test set + dictionary
        fold1 = Dictionary.get(str(col)+'enc'+str(0))
        fold2 = Dictionary.get(str(col)+'enc'+str(1))
        fold3 = Dictionary.get(str(col)+'enc'+str(2))
        fold4 = Dictionary.get(str(col)+'enc'+str(3))
        fold5 = Dictionary.get(str(col)+'enc'+str(4))
        fold6 = Dictionary.get(str(col)+'enc'+str(5))
        fold7 = Dictionary.get(str(col)+'enc'+str(6))
        fold8 = Dictionary.get(str(col)+'enc'+str(7))
        fold9 = Dictionary.get(str(col)+'enc'+str(8))
        fold10 = Dictionary.get(str(col)+'enc'+str(9))       
        
        #applying mean of all folds to test dataset
        folds = pd.concat([fold1, fold2, fold3, fold4, fold5, fold6,fold7, fold8, fold9, fold10], axis=1)
        '''
        print('printing folds')
        print(folds)
        '''
        foldsMean = folds.mean(axis=1)                
        #colMean = trainCopy.groupby(col)[target].mean()
        testCopy[col+'Enc'] = testCopy[col].map(foldsMean)
            
    trainCopy = trainCopy.filter(like = 'Enc', axis=1)
    '''
    print('trainCopy')
    print(trainCopy)
    '''
    trainCopy['binaryMean'] = trainCopy.mean(axis=1)
    trainCopy['binaryMin'] = trainCopy.min(axis=1)
    trainCopy['binaryMax'] = trainCopy.max(axis=1)
    '''
    print('printingMean')
    print(trainCopy['binaryMean'])
    print('printingMax')
    print(trainCopy['binaryMax'])
    '''
    testCopy = testCopy.filter(like = 'Enc', axis=1)
    testCopy['binaryMean'] = testCopy.mean(axis=1)
    testCopy['binaryMin'] = testCopy.min(axis=1)
    testCopy['binaryMax'] = testCopy.max(axis=1)

    trainCopy = trainCopy.filter(like = 'binary', axis=1)
    testCopy = testCopy.filter(like = 'binary', axis=1)
    
    return trainCopy, testCopy
    


def plotLabels (ax, spacing =4):
    for bar in ax.patches:
        yValue = bar.get_height()
        xValue = bar.get_x() + bar.get_width()/2
        
        va='bottom'
        
        if yValue <0:
            spacing=-1
            va='top'
        label ='{:.3f}'.format(yValue)
        
        ax.annotate(label, (xValue, yValue), xytext=(0,spacing), textcoords='offset points', ha='center', va=va)

def histHelper (ax, data, x, hue):
    xValues = data[x]
    dataHue = data[hue]
    hueLabels = sorted(dataHue.unique())
    
    
    #plotting
    values = []
    color = sns.color_palette(palette='icefire', n_colors=len(hueLabels))
    
    for i in hueLabels:
        try:
            add = np.array(dataHue == i)
            values.append(np.array(xValues[add]))
            
        except KeyError:
            values.append(np.array([]))
            
    ax.hist(x = values, color = color, bins=10, histtype='barstacked', label=hueLabels)
    ax.legend()
    ax.set(xlabel=x)
    
def pdfHist(fileName, data, items):
    plt.close("all")
    
    with PdfPages(fileName) as pdf:
        nRows, nCols = 2 , 2
        axes = nRows*nCols
        
        for i, z in enumerate(items):
            nAxis = i % axes
            r = nAxis// nRows
            c = nAxis % nRows

            if nAxis == 0:
                f, ax = plt.subplots(nRows, nCols, figsize = (15,10))
                
            histHelper(ax[r,c], data, z, 'label')
                
            print('drawing{}'.format(z))
                
            if (i % axes == axes-1) or (i == len(items)-1):
                f.tight_layout()
                pdf.savefig(f)
                plt.close('all')
                print('pdf histplot saved')
                
            
def scatterHelper(ax, data, x, y, hue):
    xValues = data[x]
    yValues = data[y]
    hueValues = data[hue]
    hueLabels=sorted(hueValues.unique())
    
    #random sample with set seed
    np.random.seed(0)
    length = len(data)
    sampleSelection = np.random.choice(length, np.min([5000,length]), replace=False)
    
    xSample = xValues[sampleSelection]
    ySample = yValues[sampleSelection]
    hueSample = hueValues[sampleSelection]
    #plotting
    for i, z in enumerate(hueLabels):
        try:
            add = np.array(hueSample == z)
            labelSampleX = xSample[add]
            labelSampleY = ySample[add]
            ax.scatter(labelSampleX, labelSampleY, s=10, color=sns.color_palette(palette='icefire', n_colors=len(hueLabels))[i], alpha = .8, marker='+', edgecolors='none', label = z, rasterized = True)
        except KeyError:
            print("Key error {}".format(z))
            
    ax.legend()
    ax.set(xlabel=x, ylabel=y)
    
def pdfScatter(fileName, data, items):
    plt.close("all")
    
    with PdfPages(fileName) as pdf:
        nRows, nCols = 2 , 2
        axes = nRows*nCols
        
        for i, z in enumerate(items):
            nAxis = i % axes
            r = nAxis// nRows
            c = nAxis % nRows

            if nAxis == 0:
                f, ax = plt.subplots(nRows, nCols, figsize = (15,10))
                
            x, y = z
            scatterHelper(ax[r,c], data, x, y, 'label')
                
            print('drawing{}'.format(z))
                
            if (i % axes == axes-1) or (i == len(items)-1):
                f.tight_layout()
                pdf.savefig(f)
                plt.close('all')
                print('pdf scatter saved')   
                
                
        
                
    
    
 
def hypParamTest(trainInput, trainOutput, predictors):
    '''
    depthList = pd.Series([2,2.5,3,3.5,4,4.5,5])
    
    accuracies = depthList.map(lambda x: (model_selection.cross_val_score(GradientBoostingRegressor(max_depth = x), trainInput.loc[:, predictors], trainOutput, cv=10, scoring='r2')).mean())
    print(accuracies)
    
    
    plt.plot(depthList, accuracies)
    plt.xlabel('Depth')
    plt.ylabel('Accuracy')
    
    
    print("Most Efficient: ", depthList.loc[accuracies.idxmax()])
    '''
    
    '''
    alphaList = pd.Series([50,100,150,200,250,300])
    
    accuracies = alphaList.map(lambda x: (model_selection.cross_val_score(Ridge(alpha = x), trainInput.loc[:, predictors], trainOutput, cv=10, scoring='r2')).mean())
    print(accuracies)
    
    
    plt.plot(alphaList, accuracies)
    plt.xlabel('Alpha')
    plt.ylabel('Accuracy')
    
    
    print("Most Efficient: ", alphaList.loc[accuracies.idxmax()])
    '''
    alphaList = pd.Series([50,100,200,300,400,500])
    
    accuracies = alphaList.map(lambda x: (model_selection.cross_val_score(Lasso(alpha = x), trainInput.loc[:, predictors], trainOutput, cv=10, scoring='r2')).mean())
    print(accuracies)
    
    
    plt.plot(alphaList, accuracies)
    plt.xlabel('Alpha')
    plt.ylabel('Accuracy')
    
    
    print("Most Efficient: ", alphaList.loc[accuracies.idxmax()])
    

'''
Does k-fold CV on the Kaggle training set using LinearRegression.
(You might review the discussion in hw09 about the so-called "Kaggle training set"
versus other sets.)
'''
def doExperiment(trainInput, trainOutput, predictors):
    alg = LinearRegression()

    cvMeanScore = model_selection.cross_val_score(alg, trainInput.loc[:, predictors], trainOutput, cv=10, scoring='r2', n_jobs=-1).mean()
    #cvMeanScore = model_selection.cross_val_score(alg, trainInput.loc[:, predictors], trainOutput, cv=10, scoring='accuracy', n_jobs=-1).mean()

    print("CV Average Score:", cvMeanScore)
    
    
    '''
    #We added GradientBoostingRegressor algorithm within the doExperiment Function
    '''
    
    alg1 = GradientBoostingRegressor()
    alg1.fit(trainInput.loc[:,predictors], trainOutput)
    cvScores = model_selection.cross_val_score(alg1, trainInput.loc[:, predictors], trainOutput, cv=10, scoring='r2')
    print("Accuracy: ", cvScores.mean())
    
    '''
    #We added Ridge Regression
    '''
    alg2 = Ridge(alpha=250)
    alg2.fit(trainInput.loc[:,predictors], trainOutput)
    cvScores = model_selection.cross_val_score(alg2, trainInput.loc[:, predictors], trainOutput, cv=10, scoring='r2')
    print("Accuracy: ", cvScores.mean())

    '''
    #We added Lasso Regression
    '''
    alg3 = Lasso(alpha=800)
    alg3.fit(trainInput.loc[:,predictors], trainOutput)
    cvScores = model_selection.cross_val_score(alg3, trainInput.loc[:, predictors], trainOutput, cv=10, scoring='r2')
    print("Accuracy: ", cvScores.mean())
    
    '''
    #We added Bagging meta-estimator
    '''
    alg4 = BaggingRegressor(KNeighborsClassifier(), max_samples = 0.85, max_features=0.85)
    alg4.fit(trainInput.loc[:,predictors], trainOutput)
    cvScores = model_selection.cross_val_score(alg4, trainInput.loc[:, predictors], trainOutput, cv=10, scoring='r2')
    print("Accuracy: ", cvScores.mean())
    
    
# ===============================================================================
'''
Runs the algorithm on the testing set and writes the results to a csv file.
'''
def doKaggleTest(trainInput, testInput, trainOutput, testIDs, predictors):
    '''
    alg = LinearRegression()

    # Train the algorithm using all the training data
    alg.fit(trainInput.loc[:, predictors], trainOutput)

    # Make predictions on the test set.
    predictions = alg.predict(testInput.loc[:, predictors])

    # Create a new dataframe with only the columns Kaggle wants from the dataset.
    submission = pd.DataFrame({
        "Id": testIDs,
        "SalePrice": predictions
    })

    # Prepare CSV
    submission.to_csv('data/testResults.csv', index=False)
    # Now, this .csv file can be uploaded to Kaggle
    '''
    
    '''
    WE ARE USING GRADIENTBOOSTINGREGRESSOR AS IT GAVE THE BEST PREDICTIONS FOR OUR ANALYSIS
    '''
    
    alg = GradientBoostingRegressor()

    # Train the algorithm using all the training data
    alg.fit(trainInput.loc[:, predictors], trainOutput)

    # Make predictions on the test set.
    predictions = alg.predict(testInput.loc[:, predictors])

    # Create a new dataframe with only the columns Kaggle wants from the dataset.
    submission = pd.DataFrame({
        "Id": testIDs,
        "SalePrice": predictions
    })

    # Prepare CSV
    submission.to_csv('data/testResults.csv', index=False)
    # Now, this .csv file can be uploaded to Kaggle
# ============================================================================
# Data cleaning - conversion, normalization

'''
Pre-processing code will go in this function (and helper functions you call from here).
'''
def transformData(trainDF, testDF):
    
    fullDF = trainDF
    trainInput = trainDF.iloc[:, :80]
    testInput = testDF.iloc[:, :]
    
   
    
    trainOutput = trainDF.loc[:, 'SalePrice']
    testIDs = testDF.loc[:, 'Id']
    
    
    
    
    'start preprocessing'
    
    '''
    We ran correlation test with all existing numerical columns and dropped all columns that had correlations between - 0.2
    and +0.3. This is because, we believe that such small correlation is not a good predictor for sales price.
    
    '''
    trainInput = trainInput.drop(['LotArea','OverallCond','BsmtUnfSF','BsmtFullBath','BsmtHalfBath','YrSold'], axis=1)
    testInput = testInput.drop(['LotArea','OverallCond','BsmtUnfSF','BsmtFullBath','BsmtHalfBath','YrSold'], axis=1)
    
    '''
    After inspecting the dataset we noticed that PoolQC, MiscVal attributes were almost entirely 'NA'. Hence, it does not
    provide anything valuable for our analysis. Therefore, we dropped it.
    We also realized that RoofMatl attribute had mostly 'CompShg' values. Following the same logic, it is not useful
    for our comparison. We dropped it.
    '''
    trainInput = trainInput.drop(['PoolQC','MiscVal','RoofMatl'], axis =1)
    testInput = testInput.drop(['PoolQC','MiscVal','RoofMatl'], axis =1)   
    
     #print(indCorr(fullDF, 'PoolQC'))
    
    #print(indCorr(fullDF, 'MiscVal')) faarik
    #print(indCorr(fullDF, 'RoofMatl'))
    #print(indCorr(fullDF, 'Alley'))

    '''
    We classified and checked correlation with SalePrice for Fence, it was too low so we dropped it.
    
    fullDF.loc[:,'Fence'].fillna(0, inplace=True)
    fullDF.loc[:,'Fence'] = fullDF.loc[:,'Fence'].map(lambda val: 3 if(val=='GdPrv') else val)
    fullDF.loc[:,'Fence'] = fullDF.loc[:,'Fence'].map(lambda val: 3 if(val=='GdWo') else val)
    fullDF.loc[:,'Fence'] = fullDF.loc[:,'Fence'].map(lambda val: 2 if(val== 'MnPrv') else val)
    fullDF.loc[:,'Fence'] = fullDF.loc[:,'Fence'].map(lambda val: 2 if(val== 'MnWw') else val)
    fullDF.loc[:,'Fence'] = fullDF.loc[:,'Fence'].map(lambda val: 0 if(val== 'NA') else val)
    print(indCorr(fullDF, 'Fence')) 
    '''
    trainInput.drop('Fence', axis=1)
    testInput.drop('Fence', axis=1)
    
    
    '''
    MasVnrArea has missing values but has high correlation with SalePrice. We are going to fill the missing ones.
    LotFrontage also has missing values but has high correlation with SalePrice. We are going to fill the missing ones.
    '''
    
    trainInput.loc[:, 'MasVnrArea'].fillna(method='bfill', inplace=True)
    testInput.loc[:,'MasVnrArea'].fillna(method='bfill', inplace=True)
    
    trainInput.loc[:, 'LotFrontage'].fillna(method='bfill', inplace=True)
    testInput.loc[:,'LotFrontage'].fillna(method='bfill', inplace=True)
      
    '''
    Garage Cars and BsmtFinSF1 have missing values, we need to fill them before we can use as predictors.
    '''
    trainInput.loc[:, 'GarageCars'].fillna(trainInput.loc[:, 'GarageCars'].mean(), inplace=True)
    testInput.loc[:, 'GarageCars'].fillna(trainInput.loc[:, 'GarageCars'].mean(), inplace=True)
    
    trainInput.loc[:, 'BsmtFinSF1'].fillna(trainInput.loc[:, 'BsmtFinSF1'].mean(), inplace=True)
    testInput.loc[:, 'BsmtFinSF1'].fillna(trainInput.loc[:, 'BsmtFinSF1'].mean(), inplace=True)
    
    
    

    
    '''
    Column PoolArea has values that are predominately equal 
    to zero. This does not provide a good indication to the algorithm in making predictions. 
    
    '''
    trainInput = trainInput.drop('PoolArea', axis =1)
    testInput = testInput.drop('PoolArea', axis =1)    
    
    '''
    Standardized 
    
    '''
    standardizeCols = ['OverallQual','YearBuilt','YearRemodAdd','MasVnrArea','BsmtFinSF1','2ndFlrSF','1stFlrSF','GrLivArea','Fireplaces','GarageYrBlt','GarageCars','GarageArea','WoodDeckSF','OpenPorchSF','LotFrontage']
    standardize(trainInput, standardizeCols )
    standardize(testInput, standardizeCols)
    
    
    predictors = ['OverallQual', 'YearBuilt','YearRemodAdd','2ndFlrSF','1stFlrSF','GrLivArea','Fireplaces','WoodDeckSF','OpenPorchSF','BsmtCond','YearsOld','GrgYrsOld','BsmtFinType1','OpenPorchSF','HouseStyle', 'ExterQual', 'HeatingQC','BsmtCond', 'YearsOld', 'GrgYrsOld','BsmtFinType1','OpenPorchSF','BsmtFinSF1','MasVnrArea','YearRemodAdd','GarageCars','HouseStyle', 'ExterQual', 'HeatingQC']
    
    
    
    '''
    Day 2
    '''
    '''
    MSZONING
    '''
   
   
    '''
    values = fullDF.loc[:,"MSZoning"].value_counts().values
    mylabels = fullDF.loc[:,"MSZoning"].unique()
    mycolors = ['lightblue', 'lightsteelblue','silver','red','gold']
    myexplode = (0.1,0,0,0,0)
    plt.pie(values, labels = mylabels, autopct='%1.1f%%', startangle = 15, shadow = True, colors = mycolors, explode = myexplode)
    plt.title('MSZoning')
    plt.axis('equal')
    plt.show()
    '''
    
    '''
    After creating a pie chart of MSZONING values, we realized that only .8% of the values are other than RL, RM, and C. Therefore, we are classifying by grouping together some 
    of the other values. Next, we will check correlation between these and Sale Price to see the relevance of MSZoning in our analysis.
    '''
    
    
    
    '''
    fullDF.loc[:, "MSZoning"] = fullDF.loc[:, "MSZoning"].map(lambda v: 0 if v=="RL" else v)
    fullDF.loc[:, "MSZoning"] = fullDF.loc[:, "MSZoning"].map(lambda v: 1 if v=="RM" else v)
    fullDF.loc[:, "MSZoning"] = fullDF.loc[:, "MSZoning"].map(lambda v: 2 if v=="C (all)" else v)
    fullDF.loc[:, "MSZoning"] = fullDF.loc[:, "MSZoning"].map(lambda v: 3 if v=="FV" else v)
    fullDF.loc[:, "MSZoning"] = fullDF.loc[:, "MSZoning"].map(lambda v: 3 if v=="RH" else v)
    
    print(indCorr(fullDF, 'MSZoning'))
    '''
    
    '''
    After classifying MSZoning (As seen above), we got a corr figure of -0.113, which is not significant enough for our analysis. Thus, we are choosing
    to drop it.
    
    
    trainInput.loc[:, "MSZoning"] = trainInput.loc[:, "MSZoning"].map(lambda v: 0 if v=="RL" else v)
    trainInput.loc[:, "MSZoning"] = trainInput.loc[:, "MSZoning"].map(lambda v: 1 if v=="RM" else v)
    trainInput.loc[:, "MSZoning"] = trainInput.loc[:, "MSZoning"].map(lambda v: 2 if v=="C (all)" else v)
    trainInput.loc[:, "MSZoning"] = trainInput.loc[:, "MSZoning"].map(lambda v: 3 if v=="FV" else v)
    trainInput.loc[:, "MSZoning"] = trainInput.loc[:, "MSZoning"].map(lambda v: 3 if v=="RH" else v)
    '''
    
    trainInput.drop('MSZoning', axis = 1)
    testInput.drop('MSZoning', axis = 1)
    

    '''
    BSMT Qual and BSMT Condition
    '''
    
    '''
    BsmtQual and BsmtCond have the same variable names. Consequently, we will be classifying them to figure out any correlation between the two
    Both have NaN values, thus we will start by filling them so they are identifiable
    
    
    fullDF.loc[:, 'BsmtQual'] = fullDF.loc[:, 'BsmtQual'].fillna('None')
    fullDF.loc[:, 'BsmtCond'] = fullDF.loc[:, 'BsmtCond'].fillna('None')
    
    values = fullDF.loc[:,"BsmtQual"].value_counts().values
    mylabels = fullDF.loc[:,"BsmtQual"].unique()
    mycolors = ['lightblue', 'lightsteelblue','silver','red','gold']
    myexplode = (0.1,0,0,0,0)
    plt.pie(values, labels = mylabels, autopct='%1.1f%%', startangle = 45, shadow = True, colors = mycolors, explode = myexplode)
    plt.title('BsmtQual')
    plt.axis('equal')
    plt.show()
    
    values = fullDF.loc[:,"BsmtCond"].value_counts().values
    mylabels = fullDF.loc[:,"BsmtCond"].unique()
    mycolors = ['lightblue', 'lightsteelblue','silver','red','gold']
    myexplode = (0.1,0,0,0,0)
    plt.pie(values, labels = mylabels, autopct='%1.1f%%', startangle = 45, shadow = True, colors = mycolors, explode = myexplode)
    plt.title('BsmtCond')
    plt.axis('equal')
    plt.show()
    
    
    Though the variables are the same, there is a disparity in frequency as shown by the charts. Next, we want to do a Correlation test to dig deeper
    Additionally, Fa and None both have very small frequncies in both attributes, thus we are combining them.
    
    
    
    fullDF.loc[:,'BsmtQual'] = fullDF.loc[:,'BsmtQual'].map(lambda val: 1 if(val=='TA') else val)
    fullDF.loc[:,'BsmtQual'] = fullDF.loc[:,'BsmtQual'].map(lambda val: 2 if(val=='Gd') else val)
    fullDF.loc[:,'BsmtQual'] = fullDF.loc[:,'BsmtQual'].map(lambda val: 3 if(val== 'Ex') else val)
    fullDF.loc[:,'BsmtQual'] = fullDF.loc[:,'BsmtQual'].map(lambda val: 4 if(val== 'Fa') else val)
    fullDF.loc[:,'BsmtQual'] = fullDF.loc[:,'BsmtQual'].map(lambda val: 4 if(val==  'None') else val)
    fullDF.loc[:,'BsmtQual'] = fullDF.loc[:,'BsmtQual'].map(lambda val: 5 if(val== 'Po') else val)
    
    

    
    fullDF.loc[:,'BsmtCond'] = fullDF.loc[:,'BsmtCond'].map(lambda val: 1 if(val=='TA') else val)
    fullDF.loc[:,'BsmtCond'] = fullDF.loc[:,'BsmtCond'].map(lambda val: 2 if(val=='Gd') else val)
    fullDF.loc[:,'BsmtCond'] = fullDF.loc[:,'BsmtCond'].map(lambda val: 3 if(val== 'Ex') else val)
    fullDF.loc[:,'BsmtCond'] = fullDF.loc[:,'BsmtCond'].map(lambda val: 4 if(val== 'Fa') else val)
    fullDF.loc[:,'BsmtCond'] = fullDF.loc[:,'BsmtCond'].map(lambda val: 4 if(val==  'None') else val)
    fullDF.loc[:,'BsmtCond'] = fullDF.loc[:,'BsmtCond'].map(lambda val: 5 if(val== 'Po') else val)
    
    
    print(fullDF.loc[:, 'BsmtCond'].corr(fullDF.loc[:,'BsmtQual']))
    
    '''
    
    '''
    The Correlation Test showed a strong correlation between BsmtCond and BsmtQual, we are choosing to stick with BsmtCond, as it increased the accuracy
    of our prediction. Additionally, we are leaving BsmtCond discretized.
    '''

  
    testInput.loc[:, 'BsmtCond'] = testInput.loc[:, 'BsmtCond'].fillna('None')
    
    
    trainInput.loc[:, 'BsmtCond'] = trainInput.loc[:, 'BsmtCond'].fillna('None')
    
    testInput.loc[:,'BsmtCond'] = testInput.loc[:,'BsmtCond'].map(lambda val: 1 if(val=='TA') else val)
    testInput.loc[:,'BsmtCond'] = testInput.loc[:,'BsmtCond'].map(lambda val: 2 if(val=='Gd') else val)
    testInput.loc[:,'BsmtCond'] = testInput.loc[:,'BsmtCond'].map(lambda val: 3 if(val== 'Ex') else val)
    testInput.loc[:,'BsmtCond'] = testInput.loc[:,'BsmtCond'].map(lambda val: 4 if(val== 'Fa') else val)
    testInput.loc[:,'BsmtCond'] = testInput.loc[:,'BsmtCond'].map(lambda val: 4 if(val==  'None') else val)
    testInput.loc[:,'BsmtCond'] = testInput.loc[:,'BsmtCond'].map(lambda val: 4 if(val== 'Po') else val)
    
    trainInput.loc[:,'BsmtCond'] = trainInput.loc[:,'BsmtCond'].map(lambda val: 1 if(val=='TA') else val)
    trainInput.loc[:,'BsmtCond'] = trainInput.loc[:,'BsmtCond'].map(lambda val: 2 if(val=='Gd') else val)
    trainInput.loc[:,'BsmtCond'] = trainInput.loc[:,'BsmtCond'].map(lambda val: 3 if(val== 'Ex') else val)
    trainInput.loc[:,'BsmtCond'] = trainInput.loc[:,'BsmtCond'].map(lambda val: 4 if(val== 'Fa') else val)
    trainInput.loc[:,'BsmtCond'] = trainInput.loc[:,'BsmtCond'].map(lambda val: 4 if(val==  'None') else val)
    trainInput.loc[:,'BsmtCond'] = trainInput.loc[:,'BsmtCond'].map(lambda val: 4 if(val== 'Po') else val)
    
   
    
    trainInput.drop('BsmtQual', axis = 1)
    testInput.drop('BsmtQual', axis = 1)
    
    
    
    '''
    
    BsmtFinType1 and BsmtFinType2
    
    Next, we noticed that BsmtFinType1 and BsmtFinType2 shared the exact same variables as well. 
    We will classify both and check correlation to see if both are relevant to our prediction
    We also noticed that both values have significnat missing values, thus we decided to fill them with the mode
    '''
    '''
    
    fullDF.loc[:,'BsmtFinType1'].fillna(fullDF.loc[:,'BsmtFinType1'].mode())
    fullDF.loc[:,'BsmtFinType2'].fillna(fullDF.loc[:,'BsmtFinType2'].mode())
    
    
    fullDF.loc[:,'BsmtFinType1'] = fullDF.loc[:,'BsmtFinType1'].map(lambda val: 1 if(val=='GLQ') else val)
    fullDF.loc[:,'BsmtFinType1'] = fullDF.loc[:,'BsmtFinType1'].map(lambda val: 2 if(val=='ALQ') else val)
    fullDF.loc[:,'BsmtFinType1'] = fullDF.loc[:,'BsmtFinType1'].map(lambda val: 3 if(val== 'BLQ') else val)
    fullDF.loc[:,'BsmtFinType1'] = fullDF.loc[:,'BsmtFinType1'].map(lambda val: 4 if(val== 'LwQ') else val)
    fullDF.loc[:,'BsmtFinType1'] = fullDF.loc[:,'BsmtFinType1'].map(lambda val: 5 if(val==  'NA') else val)
    fullDF.loc[:,'BsmtFinType1'] = fullDF.loc[:,'BsmtFinType1'].map(lambda val: 6 if(val== 'Rec') else val)
    fullDF.loc[:,'BsmtFinType1'] = fullDF.loc[:,'BsmtFinType1'].map(lambda val: 7 if(val== 'Unf') else val)
    

    
    fullDF.loc[:,'BsmtFinType2'] = fullDF.loc[:,'BsmtFinType2'].map(lambda val: 1 if(val=='GLQ') else val)
    fullDF.loc[:,'BsmtFinType2'] = fullDF.loc[:,'BsmtFinType2'].map(lambda val: 2 if(val=='ALQ') else val)
    fullDF.loc[:,'BsmtFinType2'] = fullDF.loc[:,'BsmtFinType2'].map(lambda val: 3 if(val== 'BLQ') else val)
    fullDF.loc[:,'BsmtFinType2'] = fullDF.loc[:,'BsmtFinType2'].map(lambda val: 4 if(val== 'LwQ') else val)
    fullDF.loc[:,'BsmtFinType2'] = fullDF.loc[:,'BsmtFinType2'].map(lambda val: 5 if(val==  'NA') else val)
    fullDF.loc[:,'BsmtFinType2'] = fullDF.loc[:,'BsmtFinType2'].map(lambda val: 6 if(val== 'Rec') else val)
    fullDF.loc[:,'BsmtFinType2'] = fullDF.loc[:,'BsmtFinType2'].map(lambda val: 7 if(val== 'Unf') else val)
    
    print(indCorr(fullDF, 'BsmtFinType1'))
    print(indCorr(fullDF, 'BsmtFinType2'))
    '''
    
    '''
    Doing our correlation test, BsmtFinType1 and BsmtFinType2 yielded a -0.2687 and 0.0415 correlation figure respectively. The second one is lower than
    our established threshold. Therefore, we decided to drop BsmtFinType2.
    '''
    
    testInput.loc[:,'BsmtFinType1'].fillna(method='bfill', inplace=True)
    trainInput.loc[:,'BsmtFinType1'].fillna(method='bfill', inplace=True)
    
    testInput.loc[:,'BsmtFinType1'] = testInput.loc[:,'BsmtFinType1'].map(lambda val: 1 if(val=='GLQ') else val)
    testInput.loc[:,'BsmtFinType1'] = testInput.loc[:,'BsmtFinType1'].map(lambda val: 2 if(val=='ALQ') else val)
    testInput.loc[:,'BsmtFinType1'] = testInput.loc[:,'BsmtFinType1'].map(lambda val: 3 if(val== 'BLQ') else val)
    testInput.loc[:,'BsmtFinType1'] = testInput.loc[:,'BsmtFinType1'].map(lambda val: 4 if(val== 'LwQ') else val)
    testInput.loc[:,'BsmtFinType1'] = testInput.loc[:,'BsmtFinType1'].map(lambda val: 5 if(val==  'NA') else val)
    testInput.loc[:,'BsmtFinType1'] = testInput.loc[:,'BsmtFinType1'].map(lambda val: 6 if(val== 'Rec') else val)
    testInput.loc[:,'BsmtFinType1'] = testInput.loc[:,'BsmtFinType1'].map(lambda val: 7 if(val== 'Unf') else val)
    
    trainInput.loc[:,'BsmtFinType1'] = trainInput.loc[:,'BsmtFinType1'].map(lambda val: 1 if(val=='GLQ') else val)
    trainInput.loc[:,'BsmtFinType1'] = trainInput.loc[:,'BsmtFinType1'].map(lambda val: 2 if(val=='ALQ') else val)
    trainInput.loc[:,'BsmtFinType1'] = trainInput.loc[:,'BsmtFinType1'].map(lambda val: 3 if(val== 'BLQ') else val)
    trainInput.loc[:,'BsmtFinType1'] = trainInput.loc[:,'BsmtFinType1'].map(lambda val: 4 if(val== 'LwQ') else val)
    trainInput.loc[:,'BsmtFinType1'] = trainInput.loc[:,'BsmtFinType1'].map(lambda val: 5 if(val==  'NA') else val)
    trainInput.loc[:,'BsmtFinType1'] = trainInput.loc[:,'BsmtFinType1'].map(lambda val: 6 if(val== 'Rec') else val)
    trainInput.loc[:,'BsmtFinType1'] = trainInput.loc[:,'BsmtFinType1'].map(lambda val: 7 if(val== 'Unf') else val)
    
    
    trainInput.drop('BsmtFinType2', axis = 1)
    testInput.drop('BsmtFinType2', axis = 1)
    
    '''
    YearBuilt
    '''
    
    '''
    We created a couple of additional columns to further expand and make linear the YearBuilt data.
    The DecOld variable helped categorize the years better to represent a more meaninful comparison for our algorithm
    This led to an improvement in our prediction
    '''
    

    
    trainInput.loc[:,'YearsOld'] = trainInput.loc[:, 'YearBuilt'].map(lambda x: 2021-x)
    #trainInput.loc[:,'DecOld'] = fullDF.loc[:, 'YearsOld'].map(lambda x: x//5)
    
    trainInput.loc[:,'YearsOld'].fillna(method='bfill', inplace=True)
    
    testInput.loc[:,'YearsOld'] = testInput.loc[:, 'YearBuilt'].map(lambda x: 2021-x)
    #testInput.loc[:,'DecOld'] = fullDF.loc[:, 'YearsOld'].map(lambda x: x//5)
    
    testInput.loc[:,'YearsOld'].fillna(method='bfill', inplace=True)
    
    #print(indCorr(fullDF,'YearsOld'))
    #print(indCorr(fullDF,'DecOld'))
    #print(indCorr(fullDF, 'YearBuilt'))
    
    
    '''
    GarageYrBlt
    '''
    '''
    We did the same analysis with GarageYrBlt
    '''
    
    trainInput.loc[:,'GrgYrsOld'] = trainInput.loc[:, 'GarageYrBlt'].map(lambda x: 2021-x)
    #trainInput.loc[:,'GrgDecOld'] = fullDF.loc[:, 'GarageYrBlt'].map(lambda x: x//5)
    

    trainInput.loc[:,'GrgYrsOld'].fillna(method='bfill', inplace=True)
    
    
    testInput.loc[:,'GrgYrsOld'] = fullDF.loc[:, 'GarageYrBlt'].map(lambda x: 2021-x)
    #testInput.loc[:,'GrgDecOld'] = fullDF.loc[:, 'GarageYrBlt'].map(lambda x: x//5)
    
    
    
    testInput.loc[:,'GrgYrsOld'].fillna(method='bfill', inplace=True)
    
    
    #print(testInput.loc[:,'GrgYrsOld'])
    #print(indCorr(fullDF,'GarageYrBlt'))
    #print(indCorr(fullDF,'GrgDecOld'))
    #print(indCorr(fullDF, 'GrgYrsOld'))
    
    '''
    CentralAir
    '''
    
    '''
    CentralAir attribute only had 2 values -> we want to discretize it and check correlation with SalePrice.
    During the correlation test, the correlation for Central Air was less than +0.3. Thus, it does not meet our requirements and we will drop it.
    '''  
    
    '''
    
    fullDF.loc[:,'CentralAir'] = fullDF.loc[:,'CentralAir'].map(lambda val: 1 if(val=='Y') else val)
    fullDF.loc[:,'CentralAir'] = fullDF.loc[:,'CentralAir'].map(lambda val: 0 if(val=='N') else val)
    
    print(indCorr(fullDF, 'CentralAir'))
    '''
    
    trainInput.drop('CentralAir', axis = 1)
    testInput.drop('CentralAir', axis = 1)
    
    
    '''
    The dataset has 4 porch related attributes: OpenPorchSF, EnclosedPorch, 3SsnPorch, ScreenPorch.
    We want to analyze these.
    '''
    '''
    print(indCorr(fullDF, 'OpenPorchSF'))
    print(indCorr(fullDF, 'EnclosedPorch'))
    print(indCorr(fullDF, '3SsnPorch'))
    print(indCorr(fullDF, 'ScreenPorch'))
    '''
    '''
    Next, we decided to add the attributes together, to create a total porch area variable.
    '''
    
    #fullDF.loc[:, 'ttlPorch'] = fullDF.loc[:, 'OpenPorchSF']+fullDF.loc[:, 'EnclosedPorch']+fullDF.loc[:, 'ScreenPorch']
    #print(indCorr(fullDF, 'ttlPorch'))
    
    #testInput.loc[:, 'ttlPorch'] = testInput.loc[:, 'OpenPorchSF']+testInput.loc[:, 'EnclosedPorch']+testInput.loc[:, 'ScreenPorch']
    #trainInput.loc[:, 'ttlPorch'] = trainInput.loc[:, 'OpenPorchSF']+trainInput.loc[:, 'EnclosedPorch']+trainInput.loc[:, 'ScreenPorch']
    
    '''
    The new attribute decreased our accuracy, and had a low correlation score. We will only use OpenPorchSF, as it has a correlation that is greater than
    +0.3
    '''
    trainInput = trainInput.drop(['3SsnPorch','EnclosedPorch','ScreenPorch'], axis =1)
    testInput = testInput.drop(['3SsnPorch','EnclosedPorch','ScreenPorch'], axis =1)    
    
    '''
    Next, we are looking to classify ordinal values in the Condition1 and Condition2 attributes.
    We noticed that some values relate to roads, some to railways, and some to positive features.
    We are going to classify the values accordingly.
    '''
    '''
    
    fullDF.loc[:,'Condition1'] = fullDF.loc[:,'Condition1'].map(lambda val: 0 if(val=='Artery') else val)
    fullDF.loc[:,'Condition1'] = fullDF.loc[:,'Condition1'].map(lambda val: 0 if(val=='Feedr') else val)
    fullDF.loc[:,'Condition1'] = fullDF.loc[:,'Condition1'].map(lambda val: 1 if(val=='RRNn') else val)
    fullDF.loc[:,'Condition1'] = fullDF.loc[:,'Condition1'].map(lambda val: 1 if(val=='RRAn') else val)
    fullDF.loc[:,'Condition1'] = fullDF.loc[:,'Condition1'].map(lambda val: 1 if(val=='RRNe') else val)
    fullDF.loc[:,'Condition1'] = fullDF.loc[:,'Condition1'].map(lambda val: 1 if(val=='RRAe') else val)
    fullDF.loc[:,'Condition1'] = fullDF.loc[:,'Condition1'].map(lambda val: 2 if(val=='Norm') else val)
    fullDF.loc[:,'Condition1'] = fullDF.loc[:,'Condition1'].map(lambda val: 3 if(val=='PosN') else val)
    fullDF.loc[:,'Condition1'] = fullDF.loc[:,'Condition1'].map(lambda val: 3 if(val=='PosA') else val)
    
    print(indCorr(fullDF, 'Condition1'))
    
    fullDF.loc[:,'Condition2'] = fullDF.loc[:,'Condition2'].map(lambda val: 0 if(val=='Artery') else val)
    fullDF.loc[:,'Condition2'] = fullDF.loc[:,'Condition2'].map(lambda val: 0 if(val=='Feedr') else val)
    fullDF.loc[:,'Condition2'] = fullDF.loc[:,'Condition2'].map(lambda val: 1 if(val=='RRNn') else val)
    fullDF.loc[:,'Condition2'] = fullDF.loc[:,'Condition2'].map(lambda val: 1 if(val=='RRAn') else val)
    fullDF.loc[:,'Condition2'] = fullDF.loc[:,'Condition2'].map(lambda val: 1 if(val=='RRNe') else val)
    fullDF.loc[:,'Condition2'] = fullDF.loc[:,'Condition2'].map(lambda val: 1 if(val=='RRAe') else val)
    fullDF.loc[:,'Condition2'] = fullDF.loc[:,'Condition2'].map(lambda val: 2 if(val=='Norm') else val)
    fullDF.loc[:,'Condition2'] = fullDF.loc[:,'Condition2'].map(lambda val: 3 if(val=='PosN') else val)
    fullDF.loc[:,'Condition2'] = fullDF.loc[:,'Condition2'].map(lambda val: 3 if(val=='PosA') else val)
    
    print(indCorr(fullDF, 'Condition2'))
    
    fullDF.loc[:,'CondSum'] =fullDF.loc[:,'Condition1'] + fullDF.loc[:,'Condition2']
    
    print(indCorr(fullDF, 'CondSum'))
    '''
    '''
    Despite classifying the values, we did not find a correlation even close to +0.2 with SalePrice
    Additionally, combining the two is still below a +0.2 correlation with SalePrice. We are going to drop both of these.
    '''
    
    trainInput.drop('Condition1', axis = 1)
    testInput.drop('Condition1', axis = 1)
    trainInput.drop('Condition2', axis = 1)
    testInput.drop('Condition2', axis = 1)
    
    
    '''
    Alley had many NA values, we converted them to None and checked for correlation with SalePrice. We got a high figure greater than +0.5, so 
    we are keeping it.
    '''
    '''
    fullDF.loc[:,'Alley'].fillna('None', inplace = True)
    fullDF.loc[:,'Alley'] = fullDF.loc[:,'Alley'].map(lambda val: 0 if(val=='None') else val)
    fullDF.loc[:,'Alley'] = fullDF.loc[:,'Alley'].map(lambda val: 4 if(val=='Pave') else val)
    fullDF.loc[:,'Alley'] = fullDF.loc[:,'Alley'].map(lambda val: 4 if(val== 'Grvl') else val)
    
    print(indCorr(fullDF, 'Alley'))
    
    the correlation is too low, thus we are dropping it
    '''
    testInput.drop('Alley', axis=1)
    trainInput.drop('Alley', axis=1)
    
    
    '''
    All three of: LandCountour, LotConfig, LandSlope have poor correlation with SalePrice, thus we are dropping them.
    '''
    
    testInput.drop('LandContour', axis=1)
    trainInput.drop('LandContour', axis=1)
    
    testInput.drop('LotConfig', axis=1)
    trainInput.drop('LotConfig', axis=1)
    
    testInput.drop('LandSlope', axis=1)
    trainInput.drop('LandSlope', axis=1)
    
    
    '''
    HouseStyle
    In our classification, we are coupling unfinished floors, finished floors, and the SFoyer/SLvl variables.
    
  
    fullDF.loc[:,'HouseStyle'] = fullDF.loc[:,'HouseStyle'].map(lambda val: 6 if(val=='1Story') else val)
    fullDF.loc[:,'HouseStyle'] = fullDF.loc[:,'HouseStyle'].map(lambda val: 3 if(val=='1.5Fin') else val)
    fullDF.loc[:,'HouseStyle'] = fullDF.loc[:,'HouseStyle'].map(lambda val: 4 if(val=='1.5Unf') else val)
    fullDF.loc[:,'HouseStyle'] = fullDF.loc[:,'HouseStyle'].map(lambda val: 6 if(val=='2Story') else val)
    fullDF.loc[:,'HouseStyle'] = fullDF.loc[:,'HouseStyle'].map(lambda val: 3 if(val=='2.5Fin') else val)
    fullDF.loc[:,'HouseStyle'] = fullDF.loc[:,'HouseStyle'].map(lambda val: 4 if(val=='2.5Unf') else val)
    fullDF.loc[:,'HouseStyle'] = fullDF.loc[:,'HouseStyle'].map(lambda val: 1 if(val=='SFoyer') else val)
    fullDF.loc[:,'HouseStyle'] = fullDF.loc[:,'HouseStyle'].map(lambda val: 1 if(val=='SLvl') else val)
    
    print(indCorr(fullDF, 'HouseStyle'))
    
    the correl was pretty high after classification, thus we are applying it to test and train input.
    '''
    testInput.loc[:,'HouseStyle'] = testInput.loc[:,'HouseStyle'].map(lambda val: 6 if(val=='1Story') else val)
    testInput.loc[:,'HouseStyle'] = testInput.loc[:,'HouseStyle'].map(lambda val: 3 if(val=='1.5Fin') else val)
    testInput.loc[:,'HouseStyle'] = testInput.loc[:,'HouseStyle'].map(lambda val: 4 if(val=='1.5Unf') else val)
    testInput.loc[:,'HouseStyle'] = testInput.loc[:,'HouseStyle'].map(lambda val: 6 if(val=='2Story') else val)
    testInput.loc[:,'HouseStyle'] = testInput.loc[:,'HouseStyle'].map(lambda val: 3 if(val=='2.5Fin') else val)
    testInput.loc[:,'HouseStyle'] = testInput.loc[:,'HouseStyle'].map(lambda val: 4 if(val=='2.5Unf') else val)
    testInput.loc[:,'HouseStyle'] = testInput.loc[:,'HouseStyle'].map(lambda val: 1 if(val=='SFoyer') else val)
    testInput.loc[:,'HouseStyle'] = testInput.loc[:,'HouseStyle'].map(lambda val: 1 if(val=='SLvl') else val)
    
    
    trainInput.loc[:,'HouseStyle'] = trainInput.loc[:,'HouseStyle'].map(lambda val: 6 if(val=='1Story') else val)
    trainInput.loc[:,'HouseStyle'] = trainInput.loc[:,'HouseStyle'].map(lambda val: 3 if(val=='1.5Fin') else val)
    trainInput.loc[:,'HouseStyle'] = trainInput.loc[:,'HouseStyle'].map(lambda val: 4 if(val=='1.5Unf') else val)
    trainInput.loc[:,'HouseStyle'] = trainInput.loc[:,'HouseStyle'].map(lambda val: 6 if(val=='2Story') else val)
    trainInput.loc[:,'HouseStyle'] = trainInput.loc[:,'HouseStyle'].map(lambda val: 3 if(val=='2.5Fin') else val)
    trainInput.loc[:,'HouseStyle'] = trainInput.loc[:,'HouseStyle'].map(lambda val: 4 if(val=='2.5Unf') else val)
    trainInput.loc[:,'HouseStyle'] = trainInput.loc[:,'HouseStyle'].map(lambda val: 1 if(val=='SFoyer') else val)
    trainInput.loc[:,'HouseStyle'] = trainInput.loc[:,'HouseStyle'].map(lambda val: 1 if(val=='SLvl') else val)
    
    
    '''
    RoofStyle
    
    fullDF.loc[:,'RoofStyle'] = fullDF.loc[:,'RoofStyle'].map(lambda val: 3 if(val=='Flat') else val)
    fullDF.loc[:,'RoofStyle'] = fullDF.loc[:,'RoofStyle'].map(lambda val: 2 if(val=='Gable') else val)
    fullDF.loc[:,'RoofStyle'] = fullDF.loc[:,'RoofStyle'].map(lambda val: 0 if(val=='Gambrel') else val)
    fullDF.loc[:,'RoofStyle'] = fullDF.loc[:,'RoofStyle'].map(lambda val: 2 if(val=='Hip') else val)
    fullDF.loc[:,'RoofStyle'] = fullDF.loc[:,'RoofStyle'].map(lambda val: 2 if(val=='Mansard') else val)
    fullDF.loc[:,'RoofStyle'] = fullDF.loc[:,'RoofStyle'].map(lambda val: 0 if(val=='Shed') else val)
    
    
    print(indCorr(fullDF, 'RoofStyle'))
    
    The correlation does not meet our threshold so we are dropping it.
    '''
    
    testInput.drop('RoofStyle', axis=1)
    trainInput.drop('RoofStyle', axis=1)
    
    '''
    ExterQual & ExterCond

    fullDF.loc[:,'ExterQual'] = fullDF.loc[:,'ExterQual'].map(lambda val: 5 if(val=='Ex') else val)
    fullDF.loc[:,'ExterQual'] = fullDF.loc[:,'ExterQual'].map(lambda val: 4 if(val=='Gd') else val)
    fullDF.loc[:,'ExterQual'] = fullDF.loc[:,'ExterQual'].map(lambda val: 3 if(val=='TA') else val)
    fullDF.loc[:,'ExterQual'] = fullDF.loc[:,'ExterQual'].map(lambda val: 2 if(val=='Fa') else val)
    fullDF.loc[:,'ExterQual'] = fullDF.loc[:,'ExterQual'].map(lambda val: 1 if(val=='Po') else val)

    
    
    print(indCorr(fullDF, 'ExterQual'))
    
    fullDF.loc[:,'ExterCond'] = fullDF.loc[:,'ExterCond'].map(lambda val: 5 if(val=='Ex') else val)
    fullDF.loc[:,'ExterCond'] = fullDF.loc[:,'ExterCond'].map(lambda val: 4 if(val=='Gd') else val)
    fullDF.loc[:,'ExterCond'] = fullDF.loc[:,'ExterCond'].map(lambda val: 3 if(val=='TA') else val)
    fullDF.loc[:,'ExterCond'] = fullDF.loc[:,'ExterCond'].map(lambda val: 2 if(val=='Fa') else val)
    fullDF.loc[:,'ExterCond'] = fullDF.loc[:,'ExterCond'].map(lambda val: 1 if(val=='Po') else val)

    
    
    print(indCorr(fullDF, 'ExterCond'))
    
    There is a significant correlation of 0.6826 between ExterQual and Sale Price, thus we will keep it.
    However, there is a week correlation between ExterCond and SalePrice, thus we will not consider it as a predictor.
    '''
    
    trainInput.loc[:,'ExterQual'] = trainInput.loc[:,'ExterQual'].map(lambda val: 5 if(val=='Ex') else val)
    trainInput.loc[:,'ExterQual'] = trainInput.loc[:,'ExterQual'].map(lambda val: 4 if(val=='Gd') else val)
    trainInput.loc[:,'ExterQual'] = trainInput.loc[:,'ExterQual'].map(lambda val: 3 if(val=='TA') else val)
    trainInput.loc[:,'ExterQual'] = trainInput.loc[:,'ExterQual'].map(lambda val: 2 if(val=='Fa') else val)
    trainInput.loc[:,'ExterQual'] = trainInput.loc[:,'ExterQual'].map(lambda val: 1 if(val=='Po') else val)
    
    testInput.loc[:,'ExterQual'] = testInput.loc[:,'ExterQual'].map(lambda val: 5 if(val=='Ex') else val)
    testInput.loc[:,'ExterQual'] = testInput.loc[:,'ExterQual'].map(lambda val: 4 if(val=='Gd') else val)
    testInput.loc[:,'ExterQual'] = testInput.loc[:,'ExterQual'].map(lambda val: 3 if(val=='TA') else val)
    testInput.loc[:,'ExterQual'] = testInput.loc[:,'ExterQual'].map(lambda val: 2 if(val=='Fa') else val)
    testInput.loc[:,'ExterQual'] = testInput.loc[:,'ExterQual'].map(lambda val: 1 if(val=='Po') else val)
    

    
    '''
    Heating QC
    
    fullDF.loc[:,'HeatingQC'] = fullDF.loc[:,'HeatingQC'].map(lambda val: 5 if(val=='Ex') else val)
    fullDF.loc[:,'HeatingQC'] = fullDF.loc[:,'HeatingQC'].map(lambda val: 4 if(val=='Gd') else val)
    fullDF.loc[:,'HeatingQC'] = fullDF.loc[:,'HeatingQC'].map(lambda val: 3 if(val=='TA') else val)
    fullDF.loc[:,'HeatingQC'] = fullDF.loc[:,'HeatingQC'].map(lambda val: 2 if(val=='Fa') else val)
    fullDF.loc[:,'HeatingQC'] = fullDF.loc[:,'HeatingQC'].map(lambda val: 1 if(val=='Po') else val)
    
    print(indCorr(fullDF, 'HeatingQC'))
    
    There is a high correlation of 0.8824, thus we will do the same classification for train and testing data.
    '''
    
    trainInput.loc[:,'HeatingQC'] = trainInput.loc[:,'HeatingQC'].map(lambda val: 7 if(val=='Ex') else val)
    trainInput.loc[:,'HeatingQC'] = trainInput.loc[:,'HeatingQC'].map(lambda val: 6 if(val=='Gd') else val)
    trainInput.loc[:,'HeatingQC'] = trainInput.loc[:,'HeatingQC'].map(lambda val: 3 if(val=='TA') else val)
    trainInput.loc[:,'HeatingQC'] = trainInput.loc[:,'HeatingQC'].map(lambda val: 2 if(val=='Fa') else val)
    trainInput.loc[:,'HeatingQC'] = trainInput.loc[:,'HeatingQC'].map(lambda val: 1 if(val=='Po') else val)
    
    testInput.loc[:,'HeatingQC'] = testInput.loc[:,'HeatingQC'].map(lambda val: 7 if(val=='Ex') else val)
    testInput.loc[:,'HeatingQC'] = testInput.loc[:,'HeatingQC'].map(lambda val: 6 if(val=='Gd') else val)
    testInput.loc[:,'HeatingQC'] = testInput.loc[:,'HeatingQC'].map(lambda val: 3 if(val=='TA') else val)
    testInput.loc[:,'HeatingQC'] = testInput.loc[:,'HeatingQC'].map(lambda val: 2 if(val=='Fa') else val)
    testInput.loc[:,'HeatingQC'] = testInput.loc[:,'HeatingQC'].map(lambda val: 1 if(val=='Po') else val)
    
    
    '''
    MiscFeature
    
    In classifying MiscFeature data, we noticed that the Misc Features are few and specialized. Thus we grouped them together and compared with those
    that had none.
    fullDF.loc[:,'MiscFeature'].fillna(0, inplace=True)
    
    fullDF.loc[:,'MiscFeature'] = fullDF.loc[:,'MiscFeature'].map(lambda val: 5 if(val=='Elev') else val)
    fullDF.loc[:,'MiscFeature'] = fullDF.loc[:,'MiscFeature'].map(lambda val: 5 if(val=='Gar2') else val)
    fullDF.loc[:,'MiscFeature'] = fullDF.loc[:,'MiscFeature'].map(lambda val: 5 if(val=='Othr') else val)
    fullDF.loc[:,'MiscFeature'] = fullDF.loc[:,'MiscFeature'].map(lambda val: 5 if(val=='Shed') else val)
    fullDF.loc[:,'MiscFeature'] = fullDF.loc[:,'MiscFeature'].map(lambda val: 5 if(val=='TenC') else val)
    
    print(indCorr(fullDF, 'MiscFeature'))
    
    We still got a very low correlation thus we are dropping it.
    
    
    fullDF.loc[:,'MiscFeature'] = fullDF.loc[:,'MiscFeature'].map(lambda val: 5 if(val=='Elev') else val)
    fullDF.loc[:,'MiscFeature'] = fullDF.loc[:,'MiscFeature'].map(lambda val: 4 if(val=='Gar2') else val)
    fullDF.loc[:,'MiscFeature'] = fullDF.loc[:,'MiscFeature'].map(lambda val: 3 if(val=='Othr') else val)
    fullDF.loc[:,'MiscFeature'] = fullDF.loc[:,'MiscFeature'].map(lambda val: 2 if(val=='Shed') else val)
    fullDF.loc[:,'MiscFeature'] = fullDF.loc[:,'MiscFeature'].map(lambda val: 1 if(val=='TenC') else val)
    
    data = pd.concat([fullDF.loc[:, 'SalePrice'], fullDF.loc[:,'MiscFeature']], axis=1)
    data.plot.scatter(x='MiscFeature', y='SalePrice', ylim=(0,800000));
    
    testInput.drop('MiscFeature', axis=1)
    trainInput.drop('MiscFeature', axis=1)
    '''
    
    return trainInput, testInput, trainOutput, testIDs, predictors
    
    



    
# ===============================================================================
def standardize(inputDF, cols):
    inputDF.loc[:, cols] = (inputDF.loc[:, cols] - inputDF.loc[:, cols].mean()) /inputDF.loc[:, cols].std()
    return inputDF.loc[:, cols]
    

def indCorr(df, colName):
    return df.loc[:, colName].corr(df.loc[:, 'SalePrice'])


'''
Demonstrates some provided helper functions that you might find useful.
'''
def demonstrateHelpers(trainDF):
    print("Attributes with missing values:", getAttrsWithMissingValues(trainDF), sep='\n')
    
    numericAttrs = getNumericAttrs(trainDF)
    print("Numeric attributes:", numericAttrs, sep='\n')
    
    nonnumericAttrs = getNonNumericAttrs(trainDF)
    print("Non-numeric attributes:", nonnumericAttrs, sep='\n')

    print("Values, for each non-numeric attribute:", getAttrToValuesDictionary(trainDF.loc[:, nonnumericAttrs]), sep='\n')

# ===============================================================================
'''
Returns a dictionary mapping an attribute to the array of values for that attribute.
'''
def getAttrToValuesDictionary(df):
    attrToValues = {}
    for attr in df.columns.values:
        attrToValues[attr] = df.loc[:, attr].unique()

    return attrToValues

# ===============================================================================
'''
Returns the attributes with missing values.
'''
def getAttrsWithMissingValues(df):
    valueCountSeries = df.count(axis=0)  # 0 to count down the rows
    numCases = df.shape[0]  # Number of examples - number of rows in the data frame
    missingSeries = (numCases - valueCountSeries)  # A Series showing the number of missing values, for each attribute
    attrsWithMissingValues = missingSeries[missingSeries != 0].index
    return attrsWithMissingValues

# =============================================================================

'''
Returns the numeric attributes.
'''
def getNumericAttrs(df):
    return __getNumericHelper(df, True)

'''
Returns the non-numeric attributes.
'''
def getNonNumericAttrs(df):
    return __getNumericHelper(df, False)

def __getNumericHelper(df, findNumeric):
    isNumeric = df.applymap(np.isreal) # np.isreal is a function that takes a value and returns True (the value is real) or False
                                       # applymap applies the given function to the whole data frame
                                       # So this returns a DataFrame of True/False values indicating for each value in the original DataFrame whether it is real (numeric) or not

    isNumeric = isNumeric.all() # all: For each column, returns whether all elements are True
    attrs = isNumeric.loc[isNumeric==findNumeric].index # selects the values in isNumeric that are <findNumeric> (True or False)
    return attrs

# =============================================================================

if __name__ == "__main__":
    main()

