#%%
# 載入需要的package
import pandas as pd
import numpy as np
import seaborn as sns # 畫圖用
import matplotlib.pyplot as plt # 畫圖用
from matplotlib.pyplot import MultipleLocator # 畫圖用
from sklearn.linear_model import LinearRegression # 迴歸模型
from sklearn.metrics import mean_absolute_error #平均絕對誤差
from sklearn import metrics 
import xgboost as xgb # xgboost模型
from sklearn.model_selection import train_test_split # 分割訓練集與測試集
from sklearn.model_selection import RandomizedSearchCV # randomgridsearch 調參
# 讀檔
data = pd.read_csv("D:/LTIM/pass_timeseries.csv")
# 選取需要的欄位 (BSN、時間、點位、實際X位置、實際Y位置、實際膠長、膠寬、面積、passfail...)
df = data.iloc[:,[0,1,2,3,4,11,12,13,16,17,18,19,20,21,28,29]]

# %%
# 跑相關熱力圖
df_try = df.iloc[:,5:14]
df_try = df_try.rename(columns={'X13_注膠後實際X位置':'position_x','X14_注膠後實際Y位置':'positoion_y','X18_注膠後實際膠水長度':'length',
'X18_注膠後實際膠水寬度':'width','X17_注膠後實際點位膠水面積(pixel)':'area_pixel','X17_注膠後實際點位膠水面積(mm)':'area_mm',
'X7_點位膠水設定膠量':'set_weight','X7_點位膠水設定速度':'set_position_speed','X10_點膠速度設定值':'set_dispense_speed'})
df_corr = df_try.corr()
sns.set(font_scale=1)
sns.heatmap(df_corr, center=0, annot=True)
plt.show()

#%%
# 資料隨時間排序
df = df.sort_values(["時間"],ascending=True) 

fliter_bot = (df["位置"] == "Bot")
fliter_top = (df["位置"] == "Top")
# 將上下蓋、不同點位分成不同dataframe進行分析
bot = df[fliter_bot]
top = df[fliter_top]

bot_1 = df[(df["X3_點位號"] == 1) & (df["位置"] == "Bot")]
bot_2 = df[(df["X3_點位號"] == 2) & (df["位置"] == "Bot")]
bot_3 = df[(df["X3_點位號"] == 3) & (df["位置"] == "Bot")]
bot_4 = df[(df["X3_點位號"] == 4) & (df["位置"] == "Bot")]
bot_5 = df[(df["X3_點位號"] == 5) & (df["位置"] == "Bot")]
bot_6 = df[(df["X3_點位號"] == 6) & (df["位置"] == "Bot")]
bot_7 = df[(df["X3_點位號"] == 7) & (df["位置"] == "Bot")]
bot_8 = df[(df["X3_點位號"] == 8) & (df["位置"] == "Bot")]
bot_9 = df[(df["X3_點位號"] == 9) & (df["位置"] == "Bot")]
bot_10 = df[(df["X3_點位號"] == 10) & (df["位置"] == "Bot")]
bot_11 = df[(df["X3_點位號"] == 11) & (df["位置"] == "Bot")]
bot_12 = df[(df["X3_點位號"] == 12) & (df["位置"] == "Bot")]
bot_13 = df[(df["X3_點位號"] == 13) & (df["位置"] == "Bot")]
bot_14 = df[(df["X3_點位號"] == 14) & (df["位置"] == "Bot")]
bot_15 = df[(df["X3_點位號"] == 15) & (df["位置"] == "Bot")]
bot_16 = df[(df["X3_點位號"] == 16) & (df["位置"] == "Bot")]

top_1 = df[(df["X3_點位號"] == 1) & (df["位置"] == "Top")]
top_2 = df[(df["X3_點位號"] == 2) & (df["位置"] == "Top")]
top_3 = df[(df["X3_點位號"] == 3) & (df["位置"] == "Top")]
top_4 = df[(df["X3_點位號"] == 4) & (df["位置"] == "Top")]
top_5 = df[(df["X3_點位號"] == 5) & (df["位置"] == "Top")]
top_6 = df[(df["X3_點位號"] == 6) & (df["位置"] == "Top")]

top_all = [top_1,top_2,top_3,top_4,top_5,top_6]
bot_all = [bot_1,bot_2,bot_3,bot_4,bot_5,bot_6,bot_7,bot_8,bot_9,bot_10,bot_11,bot_12,bot_13,bot_14,bot_15,bot_16]

#%%
# 去掉重複BSN的資料
def remove_repeat(x):
    for i in range(len(x)):
        x[i].reset_index(inplace=True,drop=True)
        new_list = []
        repeat_list = []
        for j in range(len(x[i].iloc[:,0])):
            if x[i].iloc[j,0] not in new_list:
                new_list.append(x[i].iloc[j,0])
            else:
                repeat_list.append(j)
        x[i].drop(index=repeat_list, inplace=True)

remove_repeat(top_all)
remove_repeat(bot_all)


# 核對資料筆數，若有資料數量不一致，則將個別BSN做比對，挑出多出的BSN回去找照片找原因，例：

# #下1,11點位數量不一致
# b11 = bot_11.iloc[:,0:2]
# b12 = bot_12.iloc[:,0:2]
# s = b12.merge(b11, how = 'outer', indicator=True)
# y = s[(s["_merge"] == "right_only")]
# print(y)

# %%
#================================將有問題的資料去除======================================
for i in [1,2,4,5]:
    recordname = top_all[i][top_all[i]["BSN"]=="PHAX224100G83P8C"].index
    top_all[i].drop(recordname, inplace=True)
# %%
for i in range(len(bot_all)):
    if i==12:
        continue
    recordname = bot_all[i][bot_all[i]["BSN"]=="PHAX224105363P8C"].index
    bot_all[i].drop(recordname, inplace=True)
    recordname = bot_all[i][bot_all[i]["BSN"]=="PHAX224100G83P8C"].index
    bot_all[i].drop(recordname, inplace=True)

# %%
for i in range(len(top_all)):
    recordname = top_all[i][top_all[i]["BSN"]=="PHAX224101HD3P8C"].index
    top_all[i].drop(recordname, inplace=True)
    recordname = top_all[i][top_all[i]["BSN"]=="PHAX224104UU3P8C"].index
    top_all[i].drop(recordname, inplace=True)
    recordname = top_all[i][top_all[i]["BSN"]=="PHAX224104TC3P8C"].index
    top_all[i].drop(recordname, inplace=True)
# %%
#====================================畫變化圖=============================================
# 將下蓋面積隨資料時間變化圖畫出
for i in range(len(bot_all)):
    plt.figure(figsize=(23,5))
    x_value = range(len(bot_all[i].iloc[:,2]))
    y_value = bot_all[i].iloc[:,-3]
    plt.plot(x_value , y_value)
    plt.title(f"Bot-{i+1}-area(mm)", fontsize=20)
    x_major_locator = MultipleLocator(50)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    plt.savefig(f'D:/LTIM/點膠數值變化圖/Bot-{i+1}-area(mm).png')
    plt.show()
#%%
# 將上蓋面積隨資料時間變化圖畫出
for i in range(len(top_all)):
    plt.figure(figsize=(23,5))
    x_value = range(len(top_all[i].iloc[:,2]))
    y_value = top_all[i].iloc[:,-3]
    plt.plot(x_value , y_value)
    plt.title(f"Top-{i+1}-area(mm)", fontsize=20)
    x_major_locator = MultipleLocator(50)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    plt.savefig(f'D:/LTIM/點膠數值變化圖/Top-{i+1}-area(mm).png')
    plt.show()
# %%
# 將上蓋x位置隨資料時間變化圖畫出
for i in range(len(top_all)):
    plt.figure(figsize=(23,5))
    x_value = range(len(top_all[i].iloc[:,2]))
    y_value = top_all[i].iloc[:,8]
    plt.plot(x_value , y_value)
    plt.title(f"Top-{i+1}-position_x", fontsize=20)
    x_major_locator = MultipleLocator(50)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    plt.show()
# %%
# 將上蓋y位置隨資料時間變化圖畫出
for i in range(len(top_all)):
    plt.figure(figsize=(23,5))
    x_value = range(len(top_all[i].iloc[:,2]))
    y_value = top_all[i].iloc[:,9]
    plt.plot(x_value , y_value)
    plt.title(f"Top-{i+1}-position_y", fontsize=20)
    x_major_locator = MultipleLocator(50)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    plt.show()

# 由以上變化圖可知上蓋的數值變化較穩定、不容易有膠體相連問題，因此選擇先針對上蓋點位進行面積的預測分析

# %%
# 取六個點位的面積資料並將其合併
for i in range(len(top_all)):
    top_all[i].reset_index(inplace=True,drop=True)
top_1 = top_1.rename(columns={'X17_注膠後實際點位膠水面積(mm)':'top1_area_mm'})
top_2 = top_2.rename(columns={'X17_注膠後實際點位膠水面積(mm)':'top2_area_mm'})
top_3 = top_3.rename(columns={'X17_注膠後實際點位膠水面積(mm)':'top3_area_mm'})
top_4 = top_4.rename(columns={'X17_注膠後實際點位膠水面積(mm)':'top4_area_mm'})
top_5 = top_5.rename(columns={'X17_注膠後實際點位膠水面積(mm)':'top5_area_mm'})
top_6 = top_6.rename(columns={'X17_注膠後實際點位膠水面積(mm)':'top6_area_mm'})
time_top = pd.concat([top_1.iloc[:,2],top_1.iloc[:,-3],top_2.iloc[:,-3],top_3.iloc[:,-3],top_4.iloc[:,-3],top_5.iloc[:,-3],top_6.iloc[:,-3]], axis=1)
time_top.drop("時間",axis=1, inplace=True)

# 可嘗試取1250前段的資料
# time_top = time_top.iloc[:1250,:]
# 可嘗試標準化後再進行資料整理
# zscore = preprocessing.StandardScaler()
# time_top = zscore.fit_transform(time_top)
# time_top = pd.DataFrame(time_top)
# %%
# 取後面0.2的資料作為測試集，前面0.8為訓練集
top_train = time_top.iloc[:int(len(time_top)*0.8),:]
top_test = time_top.iloc[int(len(time_top)*0.8):,:]

#======================================建模前資料整理===========================================
##主要分為數據量(全部點位、只取第六點位)，以及時間區段(每6個與每12個)做訓練，總共四種情況，比較是否越多點位的數據納入訓練或是時間區段越長越能準確預測
# %%
# 每個時段長(ex:前六個時間點數值)作為訓練單位，預測下一個時間點的數值(ex:第七個時間點數值)。此部份將所有上蓋點位的數據納入訓練
def tm_series(z,period):
    length = len(z.index)-period
    timedata = []
    for i in range(length):
        timedata.append([])
        for j in range(len(z.columns)):
            for k in range(period):
                timedata[i].append(z.iloc[i+k,j])
    train = np.float64(timedata)
    return train

# %%
# train_x 每六個時間點作為訓練單位，總資料筆數為(1880-6)，欄位為6個時間點*6個點位(36)
train_x_6_all = tm_series(top_train,6)
train_x_6_all.shape
# %%
# train_y 選擇預測第六點位的數據，並從第七時間點往後取
train_y_6_all = np.float64(top_train.iloc[6:,5])
train_y_6_all.shape

# %%
# test_x 測試集後面的資料筆數(470-6)，欄位為6個時間點*6個點位(36)
test_x_6_all = tm_series(top_test,6)
test_x_6_all.shape
# %% 
# test_y 選擇預測第六點位的數據，並從測試集的第七時間點往後取
test_y_6_all = np.float64(top_test.iloc[6:,5])
test_y_6_all.shape

#%%
# 將時間段改成每12個時間點為訓練單位，總資料筆數為(1880-12)，欄位為12個時間點*6個點位(72)
train_x_12_all = tm_series(top_train,12)
train_x_12_all.shape
# %%
# train_y 選擇預測第六點位的數據，並從第13時間點往後取
train_y_12_all = np.float64(top_train.iloc[12:,5])
train_y_12_all.shape

# %%
# test_x 測試集後面的資料筆數(470-6)，欄位為12個時間點*6個點位(36)
test_x_12_all = tm_series(top_test,12)
test_x_12_all.shape
# %%
# test_y 選擇預測第六點位的數據，並從測試集的第13時間點往後取
test_y_12_all = np.float64(top_test.iloc[12:,5])
test_y_12_all.shape

#%%
# 此部份只將第六點位的數據納入訓練，以下原理同上
def tm_series_p6(z,period):
    length = len(z.index)-period
    timedata = []
    for i in range(length):
        timedata.append([])
        for k in range(period):
            timedata[i].append(z.iloc[i+k,5])
    train = np.float64(timedata)
    return train
#%%
# train_x 每六個時間點作為訓練單位，總資料筆數為(1880-6)，欄位為6個時間點*1個點位(6)
train_x_6_p6 = tm_series_p6(top_train,6)
train_x_6_p6.shape
# %%
train_y_6_p6 = np.float64(top_train.iloc[6:,5])
train_y_6_p6.shape

# %%
test_x_6_p6 = tm_series_p6(top_test,6)
test_x_6_p6.shape
# %%
test_y_6_p6 = np.float64(top_test.iloc[6:,5])
test_y_6_p6.shape

#%%
train_x_12_p6 = tm_series_p6(top_train,12)
train_x_12_p6.shape
# %%
train_y_12_p6 = np.float64(top_train.iloc[12:,5])
train_y_12_p6.shape

# %%
test_x_12_p6 = tm_series_p6(top_test,12)
test_x_12_p6.shape
# %%
test_y_12_p6 = np.float64(top_test.iloc[12:,5])
test_y_12_p6.shape
#==========================================建模==================================================
# %%
# 建立一般線性迴歸模型
def lr_model(train_x,train_y,test_x,test_y):
    lr_model = LinearRegression()
    lr_model = lr_model.fit(train_x,train_y)
    pred_y = lr_model.predict(test_x)
    MAE = np.round(metrics.mean_absolute_error(test_y, pred_y), 3) # 平均絕對誤差
    MSE = np.round(metrics.mean_squared_error(test_y, pred_y), 3) # 平均平方差
    MAPE = np.round(metrics.mean_absolute_percentage_error(test_y, pred_y)*100,3) # 相對平均誤差
    score = lr_model.score(test_x, test_y) # R^2
    # 畫模型預測數值與實際數值變化圖
    plt.figure(figsize=(20,10))
    plt.title("Linear Regression", fontsize=25)
    plt.plot(range(len(test_y)),test_y,label='test_y', color='g')
    plt.plot(range(len(test_y)),pred_y,label='pred_y', color='orange')
    plt.xlabel("Number of data", fontsize=20)
    plt.ylabel("Point 6 area(mm)", fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    x_major_locator = MultipleLocator(50) # x軸以50為間距單位
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    ax.legend(fontsize=15, loc='lower right') # 定圖例位置
    plt.show()
    # 畫誤差變化圖
    plt.figure(figsize=(20,10))
    plt.title("Linear Regression Error Line", fontsize=25)
    plt.plot(range(len(test_y)),abs(test_y-pred_y))
    plt.xlabel("Number of data", fontsize=20)
    plt.ylabel("Point 6 area(mm)", fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylim(-10,250)
    x_major_locator = MultipleLocator(50)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    plt.show()
    # 回傳結果數值
    print('R^2: ' + str(np.round(score*100,3)) + '%')
    return MAE, MSE, str(MAPE)+"%"
#%%
# 建立XGBoost模型
def xgb_model(train_x,train_y,test_x,test_y):
    xgb_model = xgb.XGBRegressor(objective = 'reg:squarederror',verbosity=2)
    xgb_model = xgb_model.fit(train_x,train_y)
    pred_y = xgb_model.predict(test_x)
    MAE = np.round(metrics.mean_absolute_error(test_y, pred_y), 3)
    MSE = np.round(metrics.mean_squared_error(test_y, pred_y), 3)
    MAPE = np.round(metrics.mean_absolute_percentage_error(test_y, pred_y)*100,3)
    score = xgb_model.score(test_x, test_y)
    # 畫模型預測數值與實際數值變化圖
    plt.figure(figsize=(20,10))
    plt.title("XGBoost", fontsize=20)
    plt.plot(range(len(test_y)),test_y,label='test_y', color='g')
    plt.plot(range(len(test_y)),pred_y,label='pred_y', color='orange')
    plt.xlabel("Number of data", fontsize=20)
    plt.ylabel("Point 6 area(mm)", fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    x_major_locator = MultipleLocator(50)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    plt.legend(fontsize=15, loc='lower right')
    plt.show()
    # 畫誤差變化圖
    plt.figure(figsize=(20,10))
    plt.title("XGBoost Error Line", fontsize=20)
    plt.plot(range(len(test_y)),abs(test_y-pred_y))
    plt.xlabel("Number of data", fontsize=20)
    plt.ylabel("Point 6 area(mm)", fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylim(-10,270)
    x_major_locator = MultipleLocator(50)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    plt.show()
    # 回傳結果數值
    print('R^2: ' + str(np.round(score*100,3)) + '%')
    return MAE, MSE, str(MAPE)+"%"

#%%
# 建立用Randomgridsearch調超參數的XGBoost模型
def xgb_model_gd(train_x,train_y,test_x,test_y):
    xgb_model = xgb.XGBRegressor(objective = 'reg:squarederror',verbosity=2)
    # 設定樹深度、學習率、樹數量等參數範圍
    hyperparameter_space = {'max_depth':[2,3,4,5,6,7,8,9], 
                        'min_child_weight':[0.5, 1, 2, 3, 4, 5, 6],
                        'gamma':[0,0.1,0.2],
                        'learning_rate':[0.01, 0.05 , 0.1, 0.02],
                        'n_estimators': [300,350,400,450,500,550,600,650,700,750,800,850,900,950,1000],
                        'subsample': [0.6, 0.7, 0.8, 0.9], 
                        'colsample_bytree': [0.3, 0.5, 0.7, 0.9],
                        'reg_alpha': [0, 0.1, 1, 2], 
                        'reg_lambda': [0.1, 1, 2]}
    # 隨機選取30種參數組合
    rs = RandomizedSearchCV(xgb_model, hyperparameter_space, n_iter=30, scoring='neg_mean_absolute_error', n_jobs=-1, cv=5, random_state=1) 
    rs.fit(train_x,train_y)
    rs.best_estimator_.fit(train_x,train_y) #選擇最優良的參數組合
    y_pred = rs.best_estimator_.predict(test_x)
    score = rs.best_estimator_.score(test_x, test_y) 
    MAE = np.round(metrics.mean_absolute_error(test_y, y_pred), 3)
    MSE = np.round(metrics.mean_squared_error(test_y, y_pred), 3)
    MAPE = np.round(metrics.mean_absolute_percentage_error(test_y, y_pred)*100,3)
    # 畫模型預測數值與實際數值變化圖
    plt.figure(figsize=(20,10))
    plt.title("XGBoost Random Gridsearch", fontsize=25)
    plt.plot(range(len(test_y)),test_y,label='test_y', color='g')
    plt.plot(range(len(test_y)),y_pred,label='pred_y', color='orange')
    plt.xlabel("Number of data", fontsize=20)
    plt.ylabel("Point 6 area(mm)", fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    x_major_locator = MultipleLocator(50)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    plt.legend(fontsize=15, loc='lower right')
    plt.show()
    # 畫誤差變化圖
    plt.figure(figsize=(20,10))
    plt.title("XGBoost RG Error Line", fontsize=20)
    plt.plot(range(len(test_y)),abs(test_y-y_pred))
    plt.ylabel("Point 6 area(mm)", fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylim(-10,250)
    x_major_locator = MultipleLocator(50)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    plt.show()
    # 回傳結果數值
    print('R^2: ' + str(np.round(score*100,3)) + '%')
    return rs.best_params_ , MAE, MSE, str(MAPE)+"%"


# %%
lr_model(train_x_6_all,train_y_6_all,test_x_6_all,test_y_6_all)
#%%
xgb_model(train_x_6_all,train_y_6_all,test_x_6_all,test_y_6_all)
#%%
# 會跑一段時間!!
xgb_model_gd(train_x_6_all,train_y_6_all,test_x_6_all,test_y_6_all)
# %%
lr_model(train_x_12_all,train_y_12_all,test_x_12_all,test_y_12_all)
#%%
xgb_model(train_x_12_all,train_y_12_all,test_x_12_all,test_y_12_all)
#%%
# 會跑一段時間!!
xgb_model_gd(train_x_12_all,train_y_12_all,test_x_12_all,test_y_12_all)
# %%
lr_model(train_x_6_p6,train_y_6_p6,test_x_6_p6,test_y_6_p6)
#%%
xgb_model(train_x_6_p6,train_y_6_p6,test_x_6_p6,test_y_6_p6)
#%%
# 會跑一段時間!!
xgb_model_gd(train_x_6_p6,train_y_6_p6,test_x_6_p6,test_y_6_p6)
# %%
lr_model(train_x_12_p6,train_y_12_p6,test_x_12_p6,test_y_12_p6)
#%%
xgb_model(train_x_12_p6,train_y_12_p6,test_x_12_p6,test_y_12_p6)
#%%
# 會跑一段時間!!
xgb_model_gd(train_x_12_p6,train_y_12_p6,test_x_12_p6,test_y_12_p6)

# %%
# 六個點位的資料分布圖，平均值、標準差
for i in range(len(time_top.columns)):
    time_top.iloc[:,i].plot(kind='hist', title=f'top{i+1}_area Distribution', bins=50)
    plt.axvline(time_top.iloc[:,i].mean(), color='red', linestyle='dashed', linewidth=2)
    plt.axvline(time_top.iloc[:,i].mean()-time_top.iloc[:,i].std(), color='magenta', linestyle='dashed', linewidth=1.5)
    plt.axvline(time_top.iloc[:,i].mean()+time_top.iloc[:,i].std(), color='magenta', linestyle='dashed', linewidth=1.5)
    plt.show()
    print (f'Mean: {time_top.iloc[:,i].mean():.3f}')
    print (f'Std: {time_top.iloc[:,i].std():.3f}')