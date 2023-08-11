# 沒時間註解直接看教學網站，https://ithelp.ithome.com.tw/articles/10206312
#%%
import pandas as pd
import numpy as np
import seaborn as sns # 畫圖用
import matplotlib.pyplot as plt # 畫圖用
from matplotlib.pyplot import MultipleLocator # 畫圖用
from sklearn.metrics import mean_absolute_error #平均絕對誤差
from sklearn import metrics 

data = pd.read_csv("D:/LTIM/pass_timeseries.csv")
df = data.iloc[:,[0,1,2,3,4,11,12,13,16,17,18,19,20,21,28,29]]
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

#%%
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
#%%
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
#%%
# 取後面0.2的資料作為測試集，前面0.8為訓練集
top_train = time_top.iloc[:int(len(time_top)*0.8),:]
top_test = time_top.iloc[int(len(time_top)*0.8):,:]
#%%
training_set = top_train.iloc[:,5:6].values
#%%
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)
#%%
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
# %%
X_train = []   #預測點的前 60 筆的資料
y_train = []   #預測點
for i in range(60, len(training_set)):  
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)  # 轉成numpy array的格式，以利輸入 RNN
#%% 
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
# %%
regressor = Sequential()
# %%
# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))
# %%
# Adding the output layer
regressor.add(Dense(units = 1))
# %%
# Compiling
regressor.compile(optimizer = 'adam', loss = 'mean_absolute_error')

# 進行訓練
regressor.fit(X_train, y_train, epochs = 50, batch_size = 32)
# %%
test_y = top_test.iloc[:, 5:6].values

#%%
dataset_total = pd.concat((top_train.iloc[:,5:6], top_test.iloc[:,5:6]), axis = 0)
#%%
inputs = dataset_total[len(dataset_total) - len(top_test)-60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs) # Feature Scaling
#%%
X_test = []
for i in range(60, (60+len(top_test))):  # timesteps一樣60； 80 = 先前的60筆資料+後面test
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))  # Reshape 成 3-dimension
# %%
y_pred = regressor.predict(X_test)
y_pred = sc.inverse_transform(y_pred)  # to get the original scale
# %%
# Visualising the results
plt.figure(figsize=(20,10))
plt.plot(test_y, color = 'g', label = 'test Y')  # 紅線表示真實面積
plt.plot(y_pred, color = 'orange', label = 'Predicted Y')  # 藍線表示預測面積
plt.title('Point 6 area Prediction',fontsize=25)
plt.xlabel("Number of data", fontsize=20)
plt.ylabel("Point 6 area(mm)", fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
x_major_locator = MultipleLocator(50)
ax = plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
plt.legend(fontsize=15, loc='lower right')
plt.show()


# %%
MAE = np.round(metrics.mean_absolute_error(test_y, y_pred), 3)
MSE = np.round(metrics.mean_squared_error(test_y, y_pred), 3)
MAPE = np.round(metrics.mean_absolute_percentage_error(test_y, y_pred)*100,3)
print(MAE, MSE, (str(MAPE)+"%"))
# %%
