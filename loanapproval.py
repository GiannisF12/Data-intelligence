import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import random

warnings.filterwarnings('ignore')

df=pd.read_csv("Training Dataset.csv")

#parakatw  ta vazoume auta gia na mhn yparxei NaN input
df.LoanAmount=df.LoanAmount.fillna(df.LoanAmount.mean())
df.Credit_History=df.Credit_History.fillna(df.Credit_History.mean())
df.Loan_Amount_Term=df.Loan_Amount_Term.fillna(df.Loan_Amount_Term.mean())
df['Gender'].fillna(df['Gender'].value_counts().idxmax(), inplace=True)
df['Married'].fillna(df['Married'].value_counts().idxmax(), inplace=True)
df.Dependents.fillna(df.Dependents.value_counts().idxmax(), inplace=True)
df.Self_Employed.fillna(df.Self_Employed.value_counts().idxmax(), inplace=True)

from sklearn.preprocessing import LabelEncoder
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error as mae,median_absolute_error as mee,classification_report as cr,accuracy_score as ac

#me ta parakatw kanoume convert tstring to float
df['Education']=LabelEncoder().fit_transform(df['Education'])
df['Dependents']=LabelEncoder().fit_transform(df['Dependents'])
df['Self_Employed']=LabelEncoder().fit_transform(df['Self_Employed'])
df['Gender']=LabelEncoder().fit_transform(df['Gender'])
df['Married']=LabelEncoder().fit_transform(df['Married'])
df['Property_Area']=LabelEncoder().fit_transform(df['Property_Area'])

from sklearn.tree import DecisionTreeClassifier 
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,median_absolute_error,classification_report,accuracy_score
model=DecisionTreeClassifier()
col=['Loan_ID','Gender','Married','CoapplicantIncome','Loan_Amount_Term','Property_Area']
df2=df
df2=df2.drop(columns=col,axis=1)

x=df2[['Dependents','Education','Self_Employed','ApplicantIncome','LoanAmount','Credit_History']]
y=df2[['Loan_Status']]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=40)
model.fit(x_train,y_train)
y_pre1 = model.predict(x_test)

from sklearn.model_selection import cross_val_score
print("accuracy:",ac(y_test,y_pre1)*100)
sco1=(cross_val_score(model,x,y,cv=5))
print("prediction:",np.mean(sco1)*100)


from sklearn.naive_bayes import GaussianNB  
classifier1 = GaussianNB()  
classifier1.fit(x_train, y_train)  

y_pred3= classifier1.predict(x_test)  

list=[1,2,3,4,5,6,7,8,9]
num = random.choice(list)
print("new input accuracy:",ac(y_test,y_pred3)*100)
sco3=(cross_val_score(classifier1,x,y,cv=num))
print("new input prediction:",np.mean(sco3)*100)

if(np.mean(sco3)*100<50):
    print("not approved")
else :
    print("approved")    