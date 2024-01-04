# 2022半導體業暑期實習成果

成果一(timeseries.py, lstm.py)

一、目的
利用機器學習方法預測點散熱膠的機台出現點膠異常之時間點，以便提早準備與維護機台減少成本消耗。

二、方法
1. 資料數約2000筆，前處理將重複的數值、有問題的資料刪除，並做EDA查驗資料分布狀態。
1. 選擇散熱膠體面積作為特徵項，將資料隨時間序列排序，將前80%的資料作為訓練集，後20%的資料作為測試集。
3. 利用滾動預測之方法做資料的轉換。
4. 選擇全部點位或特定點位以及窗口大小作為模型輸入。
5. timeseries.py選擇XGBoost模型並隨機挑選最佳超參數，lstm.py選擇lstm模型進行預測。

三、結果
1. 模型MAE=34.29, MAPE=2.26, R^2=63.13%。
2. 以全部點位以及窗口6的結果為最佳。
3. 預測值與實際值的誤差如下，可預測出趨勢且數值差距不大。
   ![image](https://github.com/YoweioY/PTI_data_analysis_intern/assets/91478099/6970fd82-9f52-4391-bca2-8f82f810654d)


成果二(passfail.py)
一、目的
利用機器學習方法判斷點膠成功與否，以輔助人工判斷。

二、方法
1. 資料數37000筆，隨機將80%的資料作為訓練集，20%的資料作為測試集。
2. 由於點膠異常(Fail)遠少於點膠成功(Pass)的資料，造成資料不平衡的問題，因此將訓練集資料做Oversampling與undersampling。
3. 利用膠體位置與面積作為input，利用SVM與Logistic迴歸預測點膠成功與否。

三、結果
利用oversampling的方法與SVM的結果最佳，F1score=98%，Accuracy=99%。
![image](https://github.com/YoweioY/PTI_data_analysis_intern/assets/91478099/e5f19ddf-22f7-4d8c-ba6e-1fe5656c4a48)


# 本成果不提供資料
