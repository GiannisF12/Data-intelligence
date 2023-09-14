from subprocess import list2cmdline
from unicodedata import name
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,ConfusionMatrixDisplay
import numpy as np
from textblob import TextBlob
 


df_data = pd.read_csv("train.csv") 


df = pd.DataFrame(df_data)
print(df.dtypes)
#df_data['tweet'] = pd.to_string(df_data['tweet'], errors='coerce')
df_data['tweet'].sort_values()
dfnew = pd.DataFrame(df_data['tweet'])
#df_data['tweet'].sort_values()

dfnew = '-'.join(str(df_data['tweet']))

df = pd.DataFrame(df_data)
print(df.dtypes)
for i in df_data['tweet']:
        blob = TextBlob(i)
        if(blob.polarity>0.0):
            df_data.loc[(df_data[df_data['tweet']==i].index), 'sentiment'] = "POSITIVE"
        elif(blob.polarity==0.0): 
            df_data.loc[(df_data[df_data['tweet']==i].index), 'sentiment'] = "NEUTRAL"
        elif(blob.polarity<0.0):
            df_data.loc[(df_data[df_data['tweet']==i].index), 'sentiment'] = "NEGATIVE"


def chat():
    global x_train,x_test,y_train,y_test,userturn
    while(1):
        userInput=""
        if(userturn==0):
            userturn=1
        if(userturn==1):
            userInput = input("User1 ('end'= NextTurn Or 'STOP'= ExitProgramm): ")
            if(userInput=="end"):
                userturn=2
            elif(userInput=="STOP"):
                break
            else:
                loggreg = LogisticRegression(solver = 'lbfgs')
                x_train=np.array(x_train).reshape(-1,1)
                x_test=np.array(x_test).reshape(-1,1) 
                loggreg.fit(x_train,y_train)
                loggreg_pre = loggreg.predict(userInput)
                print("predict:".format(loggreg_pre*100))
        elif(userturn==2):
            userInput = input("User2 ('end'= NextTurn Or 'STOP'= ExitProgramm): ")
            if(userInput=="end"):
                userturn=1
            elif(userInput=="STOP"):
                break
            else:
              loggreg = LogisticRegression(solver = 'lbfgs') 
              x_train=np.array(x_train).reshape(-1,1)
              x_test=np.array(x_test).reshape(-1,1) 
              loggreg.fit(x_train,y_train)
              loggreg_pre = loggreg.predict(userInput)
              print("predict:".format(loggreg_pre*100))

userturn=0
x = df_data['tweet']
y = df_data['sentiment']

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=1/7.0 ,random_state=0)

chat()   