import json
import numpy as np
from keras import layers, Sequential

with open("result_data/back_train.json", "r", encoding="utf-8") as f:
    import json
    data = json.load(f)

def check_correct(data):
    
    correct_predictions = 0
    center_predictions = 0
    wrong_predictions = 0
    for item in data:
        probabilities = item["probabilities"]
        high = np.argmax(list(probabilities.values()))
        print("預測:", high, "實際:", item["actual_class"])
        if high == item["actual_class"]:
            correct_predictions += 1
        if high != 1 and item["actual_class"] == 1:
            center_predictions += 1
        if high == 0 and item["actual_class"] == 2:
            wrong_predictions += 1
        if high == 2 and item["actual_class"] == 0:
            wrong_predictions += 1
    print("錯誤預測:", wrong_predictions/ len(data))
    print("正確率:", correct_predictions / len(data))
    print("中心預測率:", center_predictions / len(data))

def train_model(data, timesteps=10):
    model = Sequential([
        layers.LSTM(16, activation='relu', input_shape=(timesteps, len(data[0]["probabilities"]))),
        layers.Dense(3, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    X=[]
    pro_data = [list(item["probabilities"].values()) for item in data]
    for index in range(len(data)-timesteps):
        X.append(pro_data[index:index+timesteps])
    print(X[0])
    X = np.array(X)
    y = np.array([item["actual_class"] for item in data][10:])

    X_train, X_val = X[:int(len(X)*0.8)], X[int(len(X)*0.8):]
    y_train, y_val = y[:int(len(y)*0.8)], y[int(len(y)*0.8):]
    
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))
    
    model.save("back_edit_model.h5")
    
def edit_data(timesteps=10):
    from keras.models import load_model
    model = load_model("back_edit_model.h5")

    X=[]
    pro_data = [list(item["probabilities"].values()) for item in data]
    for index in range(len(data)-timesteps):
        X.append(pro_data[index:index+timesteps])
    
    results = model.predict(np.array(X))
    print(results)

def buy_sell(data, sold_per=0.8,threshold=0.065):
    bought = 0
    left_money = 10000
    bought_history = []
    for item in data[-7:]:
        if item["predict_class"]==2 and left_money >= item["actual_close_today"]:
            if list(item["probabilities"].values())[item["predict_class"]] > 0.9:
                newbought = left_money // item["actual_close_today"]
            elif list(item["probabilities"].values())[item["predict_class"]] < 0.9:
                newbought = int(list(item["probabilities"].values())[item["predict_class"]] * 10 **0.5)
            bought += newbought
            left_money -= newbought*item["actual_close_today"]
            all_money = left_money + bought*item["actual_close_today"]
            bought_history.append([newbought, bought, left_money, all_money, item["actual_close_today"], item["date"]])
        elif item["predict_class"]==0 and bought > 0:
            if list(item["probabilities"].values())[item["predict_class"]] > 0.9:
                newbought = bought
            elif list(item["probabilities"].values())[item["predict_class"]] < 0.9:
                newbought = int(bought * 0.5)
            bought -= newbought
            left_money += newbought * item["actual_close_today"]
            all_money = left_money + bought*item["actual_close_today"]
            bought_history.append([-newbought, bought, left_money, all_money, item["actual_close_today"], item["date"]])
        else:
            bought_history.append([0, bought, left_money, left_money + bought*item["actual_close_today"], item["actual_close_today"], item["date"]])
    all_money = left_money + bought*data[-1]["actual_close_today"]
    with open("result_data/bought_history.txt", "w", encoding="utf-8") as f:
        for item in bought_history:
            # 買入/賣出, 現有持股, 剩餘資金, 總資產, 當前價格
            f.writelines(f"{item[0]}, {item[1]}, {item[2]}, {item[3]}, {item[4]}, {item[5]}\n")
    print("總資產:", all_money)
    return all_money

def old_buy_sell(data, sold_per=0.8):
    bought = 0
    left_money = 10000
    for index in range(100):
        '''
        item_today_close = data[len(data)-100+index]["Close"]
        item_yesterday_close = data[len(data)-101+index]["Close"]

        '''
        item_today_close = data[len(data)-100+index]["actual_close_today"]
        item_yesterday_close = data[len(data)-101+index]["actual_close_today"]
        if item_today_close > item_yesterday_close:
            new_bought = left_money * sold_per // item_today_close
            bought += new_bought
            left_money -= new_bought * item_today_close
        elif item_today_close < item_yesterday_close:
            new_bought = left_money * sold_per // item_today_close
            bought -= new_bought
            left_money += new_bought * item_today_close
    all_money = left_money + bought*data[-1]["actual_close_today"]
    print("總資產:", all_money)
    return all_money

def data_lab(stock_symbol="2330.TW"):
    import yfinance as yf
    import pandas as pd

    # 抓取台積電 2330.TW 十年資料
    ticker = yf.Ticker(stock_symbol)
    data = ticker.history(period="10y", interval="1d")

    # 只取收盤價
    close_prices = data[["Close"]]

    # 存成 JSON 檔
    close_prices.to_json(stock_symbol+"_10y_close.json", orient="records", force_ascii=False, indent=4)

'''
stock_symbol = "2330.TW"
data_lab(stock_symbol)
with open(stock_symbol+"_10y_close.json", "r", encoding="utf-8") as f:
    old_buy_sell(json.load(f))
'''
'''
all_result = []
for i in range(10, 110, 10):
    print("賣出比例:", i, "%")
    all_result.append([])
    result = buy_sell(data, sold_per=i/100)
    all_result[i//10-1].append(result)
print("所有結果:", all_result)
'''
buy_sell(data, sold_per=0.4)
