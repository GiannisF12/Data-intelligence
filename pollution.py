# Programma pou dexetai dataset me idiotites edafous
# kai dinei pithanotita molunsews analoga me
# tin kathe idiothta

# Xrisimopoihthike modelo KernelRidge gia na petyxoume
# mikroteri poluplokothta toy modelou me ti xrisi
# tou alpha -> sudelestis suriknwsis

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import  mean_squared_error
from sklearn.kernel_ridge import KernelRidge

#Training kai test set apo idiotites edafous afrikis
train = pd.read_csv('train.zip')
test = pd.read_csv('test.zip')
train.head()
test.head()
train.shape
test.shape
train.isna().sum().max()
test.isna().sum().max()
train.Depth.value_counts()
train['Depth'] = train['Depth'].replace({'Topsoil' : 0 , 'Subsoil' : 1})
test.drop(['PIDN'],axis=1,inplace=True)

# spaw kai pairnw to training set gia na dw to Ca
y_ca = train['Ca']
X_ca = train.drop(['Ca','P','pH','SOC','Sand','PIDN'],axis=1)

# k is the number of features you want to select
from sklearn.feature_selection import SelectKBest, f_regression
fs = SelectKBest(score_func=f_regression,k=100)

# kanonikopoihsh toy set gia to Ca
X_can= fs.fit_transform(X_ca,y_ca)

#display the selected features
l_f = fs.get_support()
l_f2 = test.columns
l_f2 = list(l_f2)

fs_se = []
for i in range (len(l_f2)):
  if l_f[i] == True :
    fs_se.append(l_f2[i])

new_test1 = test[fs_se]

# spaw se training kai test set gia na dokimasw to modelo
x_train1,x_val1,y_train1,y_val1 = train_test_split(X_can,y_ca,random_state=42,test_size=0.2)

# kanw ena modelo regression
kr1 = KernelRidge(alpha=0.1, kernel='polynomial', degree=7, coef0=2.5)
kr1.fit(x_train1,y_train1)
y_pred1 = kr1.predict(x_val1)
mean_squared_error(y_pred1,y_val1)
pred1 = kr1.predict(new_test1)
# emfanizw pithanotita molunsis edafous me basi CALCIUM panw sto test set
# gia kathe metrisi m-xxx tou dataset to opoio einai metriseis aporofisis mesaias yperithris aktinovolias
# diladi gia kathe mikos kumatos yperithris aktinovolias bgainei mia pithanotita molunsis 0 h 1.
# epidi exoyme dataset me polla miki kumatos aytes oi pithanotites bgainoyn se pinaka me times sto [0,1].
print("Probabillity of air pollution based on Ca")
print(pred1)

# spaw kai pairnw to training set gia na dw to P
y_p = train['P']
X_p = train.drop(['Ca','P','pH','SOC','Sand','PIDN'],axis=1)
X_pn= fs.fit_transform(X_ca,y_ca)
#display the selected features
l_f = fs.get_support()
l_f2 = test.columns
l_f2 = list(l_f2)

fs_se = []
for i in range (len(l_f2)):
  if l_f[i] == True :
    fs_se.append(l_f2[i])

# spaw kai pairnw to training set gia na dw to P
new_test2 = test[fs_se]
x_train2,x_val2,y_train2,y_val2 = train_test_split(X_pn,y_p,random_state=42,test_size=0.2)
# kanw ena modelo regression
kr2 = KernelRidge(alpha=0.1, kernel='polynomial', degree=7, coef0=2.5)
kr2.fit(x_train2,y_train2)
y_pred2 = kr2.predict(x_val2)
mean_squared_error(y_pred2,y_val2)
pred2 = kr2.predict(new_test2)
# emfanizw pithanotita molunsis me basi P -> MEHLICH-3 EXTRACTABLE PHOSPHORUS panw sto test set
# gia kathe metrisi m-xxx tou dataset to opoio einai metriseis aporofisis mesaias yperithris aktinovolias
# diladi gia kathe mikos kumatos yperithris aktinovolias bgainei mia pithanotita molunsis 0 h 1.
# epidi exoyme dataset me polla miki kumatos aytes oi pithanotites bgainoyn se pinaka me times sto [0,1].
print("Probabillity of ground infection based on P")
print(pred2)
degrees = range(1,11)
for d in degrees:
  kr = KernelRidge(alpha=0.1, kernel='polynomial', degree=d, coef0=2.5)
  kr.fit(x_train2,y_train2)
  y_pred2 = kr.predict(x_val2)
  m = mean_squared_error(y_pred2,y_val2)
  # print(m)
  # bazw diafores times toy alpha gia na epirewsw ti diakumansi twn ektimisewn
  alphas = [0.1 , 0.2 ,0.3 , 0.4, 0.5 , 0.7 , 0.8 , 0.9 , 1]
  for k in alphas:
    kr = KernelRidge(alpha=k, kernel='polynomial', degree=2, coef0=2.5)
    kr.fit(x_train2,y_train2)
    y_pred2 = kr.predict(x_val2)
    m = mean_squared_error(y_pred2,y_val2)
    # print(m)

# spaw kai pairnw to training set gia na dw to PH
y_ph = train['pH']
X_ph = train.drop(['Ca','P','pH','SOC','Sand','PIDN'],axis=1)

# kanonikopoihsh
X_phn= fs.fit_transform(X_ph,y_ph)

#display the selected features
l_f = fs.get_support()
l_f2 = test.columns
l_f2 = list(l_f2)

fs_se = []
for i in range (len(l_f2)):
  if l_f[i] == True :
    fs_se.append(l_f2[i])

new_test3 = test[fs_se]

# spaw se training kai test set gia na dw to PH me to modelo
x_train3,x_val3,y_train3,y_val3 = train_test_split(X_phn,y_ph,random_state=42,test_size=0.2)
# kanw ena modelo regression
kr3 = KernelRidge(alpha=0.1, kernel='polynomial', degree=7, coef0=2.5)
kr3.fit(x_train3,y_train3)
y_pred3 = kr3.predict(x_val3)
mean_squared_error(y_pred3,y_val3)
# emfanizw pithanotita molunsis edafous me basi PH panw sto test set
# gia kathe metrisi m-xxx tou dataset to opoio einai metriseis aporofisis mesaias yperithris aktinovolias
# diladi gia kathe mikos kumatos yperithris aktinovolias bgainei mia pithanotita molunsis 0 h 1.
# epidi exoyme dataset me polla miki kumatos aytes oi pithanotites bgainoyn se pinaka me times sto [0,1].
print("Probabillity of ground infection based on PH")
pred3 = kr3.predict(new_test3)
print(pred3)

# spaw kai pairnw to training set gia na dw to SOC
y_soc = train['SOC']
X_soc = train.drop(['Ca','P','pH','SOC','Sand','PIDN'],axis=1)

X_socn= fs.fit_transform(X_soc,y_soc)

#display the selected features
l_f = fs.get_support()
l_f2 = test.columns
l_f2 = list(l_f2)

fs_se = []
for i in range (len(l_f2)):
  if l_f[i] == True :
    fs_se.append(l_f2[i])

# spaw se training kai test set gia na dw to SOC me to modelo
new_test4 = test[fs_se]
x_train4,x_val4,y_train4,y_val4 = train_test_split(X_socn,y_soc,random_state=42,test_size=0.2)
kr4 = KernelRidge(alpha=0.1, kernel='polynomial', degree=7, coef0=2.5)
kr4.fit(x_train4,y_train4)
y_pred4 = kr4.predict(x_val4)
mean_squared_error(y_pred4,y_val4)
pred4 = kr4.predict(new_test4)
# emfanizw pithanotita molunsis me basi to SOC -> SOIL ORGANIC CARBON  panw sto test set
# gia kathe metrisi m-xxx tou dataset to opoio einai metriseis aporofisis mesaias yperithris aktinovolias
# diladi gia kathe mikos kumatos yperithris aktinovolias bgainei mia pithanotita molunsis 0 h 1.
# epidi exoyme dataset me polla miki kumatos aytes oi pithanotites bgainoyn se pinaka me times sto [0,1].
print("Probabillity of air based on SOC")
print(pred4)

# spaw kai pairnw to training set gia na dw ti metavliti Sand

y_sand = train['Sand']
X_sand = train.drop(['Ca','P','pH','SOC','Sand','PIDN'],axis=1)
X_sandn= fs.fit_transform(X_sand,y_sand)
#display the selected features
l_f = fs.get_support()
l_f2 = test.columns
l_f2 = list(l_f2)

fs_se = []
for i in range (len(l_f2)):
  if l_f[i] == True :
    fs_se.append(l_f2[i])

new_test5 = test[fs_se]

# spaw se training kai test set gia na dw to SAND me to modelo
x_train5,x_val5,y_train5,y_val5 = train_test_split(X_sandn,y_sand,random_state=42,test_size=0.2)
kr5 = KernelRidge(alpha=0.1, kernel='polynomial', degree=7, coef0=2.5)
kr5.fit(x_train5,y_train5)
y_pred5 = kr5.predict(x_val5)
mean_squared_error(y_pred5,y_val5)
pred5 = kr5.predict(new_test5)
# emfanizw pithanotita molunsis me basi to Sand panw sto test set
# gia kathe metrisi m-xxx tou dataset to opoio einai metriseis aporofisis mesaias yperithris aktinovolias
# diladi gia kathe mikos kumatos yperithris aktinovolias bgainei mia pithanotita molunsis 0 h 1.
# epidi exoyme dataset me polla miki kumatos aytes oi pithanotites bgainoyn se pinaka me times sto [0,1].
print("Probabillity of pollution")
print(pred5)
