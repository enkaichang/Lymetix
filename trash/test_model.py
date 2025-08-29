import numpy as np
import pandas as pd
import yfinance as yf
import joblib
from keras.models import Model
from keras.layers import Input, MultiHeadAttention, LayerNormalization, Dropout, Dense, GlobalAveragePooling1D

# -------- 1. 載入資料，這邊用 yf 範例
ticker = "2330.TW"
start_date = "2021-12-01"
end_date = "2023-08-08"
df = yf.download(ticker, start=start_date, end=end_date)
df.dropna(inplace=True)

# -------- 2. 計算技術指標函數（請用你自己的指標計算函數）
def compute_technical_indicators(df):
    df['MA5'] = df['Close'].rolling(5).mean()
    df['MA20'] = df['Close'].rolling(20).mean()
    delta = df['Close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.rolling(14).mean()
    roll_down = down.rolling(14).mean()
    rs = roll_up / (roll_down + 1e-9)
    df['RSI14'] = 100.0 - (100.0 / (1.0 + rs))
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    ma20 = df['Close'].rolling(20).mean()
    std20 = df['Close'].rolling(20).std()
    df['BB_upper'] = ma20 + 2 * std20
    df['BB_lower'] = ma20 - 2 * std20
    low_min = df['Low'].rolling(9).min()
    high_max = df['High'].rolling(9).max()
    rsv = (df['Close'] - low_min) / (high_max - low_min + 1e-9) * 100
    df['K'] = rsv.ewm(com=2).mean()
    df['D'] = df['K'].ewm(com=2).mean()
    df['J'] = 3 * df['K'] - 2 * df['D']
    df['Volume_Change'] = df['Volume'].pct_change()
    df.dropna(inplace=True)
    return df

df = compute_technical_indicators(df)

# -------- 3. 特徵欄位，跟你模型訓練時一樣
features = ['Close', 'MA5', 'MA20', 'RSI14', 'MACD', 'MACD_signal',
            'BB_upper', 'BB_lower', 'K', 'D', 'J']

# -------- 4. 載入 scaler
scaler = joblib.load("model.pkl")

# -------- 5. 製作序列輸入
sequence_length = 180

def create_sequences(data, features, seq_len):
    X = []
    for i in range(len(data) - seq_len, len(data)):
        X.append(data[features].iloc[i-seq_len:i].values)
    return np.array(X)

X_pred = create_sequences(df, features, sequence_length)

# -------- 6. 標準化
nsamples, ntimesteps, nfeatures = X_pred.shape
X_pred_reshaped = X_pred.reshape(-1, nfeatures)
X_pred_scaled = scaler.transform(X_pred_reshaped)
X_pred_scaled = X_pred_scaled.reshape(nsamples, ntimesteps, nfeatures)

# -------- 7. 重建 Transformer 模型架構
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads)(inputs, inputs)
    x = Dropout(dropout)(x)
    x = LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

    x = Dense(ff_dim, activation="relu")(res)
    x = Dropout(dropout)(x)
    x = Dense(inputs.shape[-1])(x)
    x = LayerNormalization(epsilon=1e-6)(x)
    return x + res

inputs = Input(shape=(X_pred_scaled.shape[1], X_pred_scaled.shape[2]))
x = transformer_encoder(inputs, head_size=64, num_heads=4, ff_dim=128, dropout=0.1)
x = GlobalAveragePooling1D()(x)
x = Dropout(0.3)(x)
x = Dense(64, activation='relu')(x)
outputs = Dense(3, activation='softmax')(x)
model = Model(inputs, outputs)

# -------- 8. 載入權重（假設你有權重檔 weights.h5）
model.load_weights("model_0.1, 0.1.h5")

# -------- 9. 預測
y_pred_prob = model.predict(X_pred_scaled)
y_pred_class = np.argmax(y_pred_prob, axis=1)

label_map = {0: "跌", 1: "觀望", 2: "漲"}
#print("預測結果機率：", y_pred_prob)
print("預測類別：", [label_map[c] for c in y_pred_class])
