#%%
# 載入需要的package
import pandas as pd
import numpy as np
import seaborn as sns # 畫圖用
import matplotlib.pyplot as plt # 畫圖用
from imblearn.over_sampling import SMOTE #oversampling
from imblearn.under_sampling import NearMiss #underersampling
from sklearn.model_selection import train_test_split # 分割訓練集與測試集
from sklearn.metrics import classification_report # recall precision...報告
from sklearn.metrics import confusion_matrix # 混淆矩陣
from sklearn.linear_model import LogisticRegression # 邏輯式迴歸模型
from sklearn import svm # svm模型

# 讀檔
data = pd.read_csv("D:/LTIM/passfail.csv")
# 選取需要的欄位 (BSN、時間、點位、實際X位置、實際Y位置、實際膠長、膠寬、面積、passfail...)
df = data.iloc[:,[0,1,2,3,16,17,18,19,20,21,29,30,31]]
# 資料隨時間排序
df = df.sort_values(["時間"],ascending=True) 

# 畫圖 a=資料, b=分類欄位, c=x軸, d=y軸
def passfail_plot(a, b, c, d):
    # 畫pass跟fail的散佈圖
    plt.figure(figsize=(10,5))  
    plt.xlabel("Position_x", fontweight = "bold")                  
    plt.ylabel("Area(mm)", fontweight = "bold")   
    plt.title("Scatter of pass fail in position_X and area(mm)", fontsize = 15, fontweight = "bold")       
    sns.scatterplot(x=a.iloc[:,c], y=a.iloc[:,d], hue=b) 
    plt.legend(fontsize=10)
    plt.show()

    # 畫pass跟fail的圓餅圖
    plt.figure( figsize=(10,5) )
    b.value_counts().plot( kind='pie', colors=['lightcoral','skyblue'], autopct='%1.2f%%' )
    plt.title( 'Pass/Fail data' )  
    plt.ylabel( '' )
    plt.show()

#============================================上蓋=====================================================
# %%
# 選取上蓋資料
top_all = df[(df["位置"] == "Top")]

#%%
passfail_plot(top_all, top_all.iloc[:,-3], 4, -4)
#%%
# 資料分割 .8訓練集.2測試集，以x,y位置、面積...做特徵

train_x,test_x,train_y,test_y = train_test_split(top_all.iloc[:,[4,5,6,7,9]], top_all.iloc[:,-3], test_size=0.2, random_state=1)

# 資料不平等
# oversampling
X_re, y_re = SMOTE(random_state=42).fit_resample(train_x, train_y)
passfail_plot(X_re, y_re, 1, -1)

# undersampling
nearmiss = NearMiss (version=3)
X_train_nearmiss, y_train_nearmiss= nearmiss.fit_resample(train_x, train_y)
passfail_plot(X_train_nearmiss, y_train_nearmiss, 1, -1)

#%%
# 邏輯式迴歸模型
lr_over = LogisticRegression().fit(X_re, y_re)
y_pred_over = lr_over.predict(test_x)
print(classification_report(test_y, y_pred_over))
tn, fp, fn, tp = confusion_matrix(test_y, y_pred_over).ravel()
print(tn, fp, fn, tp)

lr_under = LogisticRegression().fit(X_train_nearmiss, y_train_nearmiss)
y_pred_under = lr_under.predict(test_x)
print(classification_report(test_y, y_pred_under))
tn, fp, fn, tp = confusion_matrix(test_y, y_pred_under).ravel()
print(tn, fp, fn, tp)
# %%
# SVM模型
# over
clf_over=svm.SVC(kernel='rbf',C=1,gamma='auto')
clf_over.fit(X_re, y_re)
clf_pred_over = clf_over.predict(test_x)
print(classification_report(test_y, clf_pred_over))
tn, fp, fn, tp = confusion_matrix(test_y, clf_pred_over).ravel()
print(tn, fp, fn, tp)

# under
clf_under=svm.SVC(kernel='rbf',C=1,gamma='auto')
clf_under.fit(X_train_nearmiss, y_train_nearmiss)
clf_pred_under = clf_under.predict(test_x)
print(classification_report(test_y, clf_pred_under))
tn, fp, fn, tp = confusion_matrix(test_y, clf_pred_under).ravel()
print(tn, fp, fn, tp)


##======================================下蓋===============================================
# %%
bot_all = df[(df["位置"] == "Bot")]
#重排索引值
bot_all.reset_index(inplace=True,drop=True)
# 將品質不良原因判斷欄位不是pass的資料作為fail資料
for i in range(len(bot_all)):
    if (bot_all.iloc[i,-2] != "pass"):
        bot_all.iloc[i,-2]="fail"

passfail_plot(bot_all, bot_all.iloc[:,-2], 4, -4)

#%%
# 資料分割 .8訓練集.2測試集，以x,y位置、面積...做特徵
train_x,test_x,train_y,test_y = train_test_split(bot_all.iloc[:,[4,5,6,7,9]], bot_all.iloc[:,-2], test_size=0.2, random_state=1)
# 資料不平等，採樣方式:
# oversampling
X_re, y_re = SMOTE(random_state=42).fit_resample(train_x, train_y)
passfail_plot(X_re, y_re, 1, -1)
# undersampling
nearmiss = NearMiss (version=3)
X_train_nearmiss, y_train_nearmiss= nearmiss.fit_resample(train_x, train_y)
passfail_plot(X_train_nearmiss, y_train_nearmiss, 1, -1)


# %%
# 邏輯式迴歸模型
lr_over = LogisticRegression().fit(X_re, y_re)
y_pred_over = lr_over.predict(test_x)
print(classification_report(test_y, y_pred_over))

lr_under = LogisticRegression().fit(X_train_nearmiss, y_train_nearmiss)
y_pred_under = lr_under.predict(test_x)
print(classification_report(test_y, y_pred_under))
# %%
# SVM模型
# over 要跑一段時間!!
clf_over=svm.SVC(kernel='rbf',C=1,gamma='auto')
clf_over.fit(X_re, y_re)
clf_pred_over = clf_over.predict(test_x)
print(classification_report(test_y, clf_pred_over))
tn, fp, fn, tp = confusion_matrix(test_y, clf_pred_over).ravel()
print(tn, fp, fn, tp)

#%%
# under
clf_under=svm.SVC(kernel='rbf',C=1,gamma='auto')
clf_under.fit(X_train_nearmiss, y_train_nearmiss)
clf_pred_under = clf_under.predict(test_x)
print(classification_report(test_y, clf_pred_under))
tn, fp, fn, tp = confusion_matrix(test_y, clf_pred_under).ravel()
print(tn, fp, fn, tp)


# %%
# 檢查錯誤判斷的fail資料中，有幾筆是品質檢測結果真的fail的資料
ypd = pd.DataFrame(test_y)
ypd['index'] = ypd.index

indexposition=[]
for i in range(len(test_y)):
    if ypd.iloc[i,0] != clf_pred_over[i]:
        indexposition.append(ypd.iloc[i,1])

real_fail = bot_all.iloc[indexposition, -3]

len(real_fail[real_fail=="fail"])/len(bot_all[bot_all.iloc[:,-3]=='fail']) # 所有真實fail資料中誤判的比例



