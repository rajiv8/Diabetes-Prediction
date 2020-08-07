#importing the dependencies
import pandas as pd
import numpy as np
import pickle

#loading the dataset
df1=pd.read_csv("diabetes-1.csv")

#dropping unnecessary feature/column
df2=df1.drop(["Pregnancies","DiabetesPedigreeFunction"],axis="columns")

#Replacing all the zero values in the column with the column mean
df2["Glucose"] = df2["Glucose"].replace(0,120.89453125)
df2["BloodPressure"] = df2["BloodPressure"].replace(0,69.10546875)
df2["SkinThickness"] = df2["SkinThickness"].replace(0,20.536458333333332)
df2["Insulin"] = df2["Insulin"].replace(0,79.79947916666667)
df2["BMI"] = df2["BMI"].replace(0,31.992578124999998)
df2["Age"] = df2["Age"].replace(0,33.240885416666664)

#splitting the dataset into train and test (20% in test)
from sklearn.model_selection import train_test_split
x = np.asarray(df2[["Glucose","BloodPressure","SkinThickness","Insulin","BMI","Age"]])
y = np.asarray(df2["Outcome"])
x_train, x_test, y_train, y_test = train_test_split( x, y, test_size=0.2, random_state=4)


# print("# Total no. of Rows {0}".format(len(df2)))
# print("# Rows missing Glucose Values {0}".format(len(df2.loc[df2.Glucose==0])))
# print("# Rows missing BloodPressure Values {0}".format(len(df2.loc[df2.BloodPressure==0])))
# print("# Rows missing Skin Thickness Values {0}".format(len(df2.loc[df2.SkinThickness==0])))
# print("# Rows missing Insulin Values {0}".format(len(df2.loc[df2.Insulin==0])))
# print("# Rows missing BMI Values {0}".format(len(df2.loc[df2.BMI==0])))
# print("# Rows missing Age Values {0}".format(len(df2.loc[df2.Glucose==0])))


#Model selection(logistic regression) and fitting the train dataset in the model
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression(C=0.01, solver='lbfgs')
LR.fit(x_train,y_train)

#predicts the target value
yhat=LR.predict(x_test)
#yhat


# model Evaluation(checking Accuracy of model)

# from sklearn.metrics import jaccard_similarity_score
# jaccard_similarity_score(y_test, yhat)

# from sklearn.metrics import confusion_matrix
# print(confusion_matrix(y_test, yhat, labels=[1,0]))

# predicts the probabolity of 0 and 1 (0 - no diabetes ,1 - diabetes)
yhat_prob = LR.predict_proba(x_test)
# yhat_prob



# print("Probability of Diabetes : {0:.2f} %".format(Diab_pred))

file = open("model.pkl","wb")
pickle.dump(LR,file)
file.close()