import os
import io
import numpy as np

def ParseReport(filename):

    LSVM_normal_all = []
    LSVM_features_all = []
    LSVM_normal_100 = []
    LSVM_features_100 = []
    LSVM_normal_10 = []
    LSVM_features_10 = []

    LR_normal_all = []
    LR_features_all = []
    LR_normal_100 = []
    LR_features_100 = []
    LR_normal_10 = []
    LR_features_10 = []

    file_opener = open(filename, "r")
    for line in file_opener:
        splitLine = line.rsplit(": ")
        #print (splitLine[0])
        #print (splitLine[-1])
        if "SVM" in splitLine[0]:
            if "entire" in splitLine[0]:
                if "pixels" in splitLine[0]:
                    LSVM_normal_all.append(float(splitLine[-1].strip("\n")))
                else:
                    LSVM_features_all.append(float(splitLine[-1].strip("\n")))
            elif "10 " in splitLine[0]:
                if "pixels" in splitLine[0]:
                    LSVM_normal_10.append(float(splitLine[-1].strip("\n")))
                else:
                    LSVM_features_10.append(float(splitLine[-1].strip("\n")))
            elif "100 " in splitLine[0]:
                if "pixels" in splitLine[0]:
                    LSVM_normal_100.append(float(splitLine[-1].strip("\n")))
                else:
                    LSVM_features_100.append(float(splitLine[-1].strip("\n")))
        elif "Logistic Regression" in splitLine[0]:
            if "entire" in splitLine[0]:
                if "pixels" in splitLine[0]:
                    LR_normal_all.append(float(splitLine[-1].strip("\n")))
                else:
                    LR_features_all.append(float(splitLine[-1].strip("\n")))
            elif "10 " in splitLine[0]:
                if "pixels" in splitLine[0]:
                    LR_normal_10.append(float(splitLine[-1].strip("\n")))
                else:
                    LR_features_10.append(float(splitLine[-1].strip("\n")))
            elif "100 " in splitLine[0]:
                if "pixels" in splitLine[0]:
                    LR_normal_100.append(float(splitLine[-1].strip("\n")))
                else:
                    LR_features_100.append(float(splitLine[-1].strip("\n")))

    print ("Mean accuracy score for Linear SVM on MNIST pixels (entire dataset): " + str(np.mean(LSVM_normal_all)) + " with standard deviation: " + str(np.std(LSVM_normal_all)))
    print ("Mean accuracy score for Linear SVM on MNIST features (entire dataset): " + str(np.mean(LSVM_features_all)) + " with standard deviation: " + str(np.std(LSVM_features_all)))
    print ("Mean accuracy score for Linear SVM on MNIST pixels (10 samples per class): " + str(np.mean(LSVM_normal_10)) + " with standard deviation: " + str(np.std(LSVM_normal_10)))
    print ("Mean accuracy score for Linear SVM on MNIST features (10 samples per class): " + str(np.mean(LSVM_features_10)) + " with standard deviation: " + str(np.std(LSVM_features_10)))
    print ("Mean accuracy score for Linear SVM on MNIST pixels (100 samples per class): " + str(np.mean(LSVM_normal_100)) + " with standard deviation: " + str(np.std(LSVM_normal_100)))
    print ("Mean accuracy score for Linear SVM on MNIST features (100 samples per class): " + str(np.mean(LSVM_features_100)) + " with standard deviation: " + str(np.std(LSVM_features_100)))

    print ("Mean accuracy score for Logistic Regression on MNIST pixels (entire dataset): " + str(np.mean(LR_normal_all)) + " with standard deviation: " + str(np.std(LR_normal_all)))
    print ("Mean accuracy score for Logistic Regression on MNIST features (entire dataset): " + str(np.mean(LR_features_all)) + " with standard deviation: " + str(np.std(LR_features_all)))
    print ("Mean accuracy score for Logistic Regression on MNIST pixels (10 samples per class): " + str(np.mean(LR_normal_10)) + " with standard deviation: " + str(np.std(LR_normal_10)))
    print ("Mean accuracy score for Logistic Regression on MNIST features (10 samples per class): " + str(np.mean(LR_features_10)) + " with standard deviation: " + str(np.std(LR_features_10)))
    print ("Mean accuracy score for Logistic Regression on MNIST pixels (100 samples per class): " + str(np.mean(LR_normal_100)) + " with standard deviation: " + str(np.std(LR_normal_100)))
    print ("Mean accuracy score for Logistic Regression on MNIST features (100 samples per class): " + str(np.mean(LR_features_100)) + " with standard deviation: " + str(np.std(LR_features_100)))


print ("********RESULTS UNTRAINED PARSED REPORT********")
ParseReport("ResultsUntrained")
print ("********RESULTS TRAINED PARSED REPORT********")
ParseReport("ResultsTrained")
