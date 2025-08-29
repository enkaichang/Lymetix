import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from keras.models import Model
from keras.layers import LSTM, Dense, Dropout, Layer, Input, Bidirectional, BatchNormalization, MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D
from keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
import joblib

SCALER_NAME = "model"
MODEL_NAME = "model_0.1, 0.1.h5"

# 技術指標函數（簡化版，無ADX）
def compute_RSI(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))

def compute_MACD(series, short=12, long=26, signal=9):
    short_ema = series.ewm(span=short, adjust=False).mean()
    long_ema = series.ewm(span=long, adjust=False).mean()
    macd = short_ema - long_ema
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def compute_BBands(df, n=20, k=2):
    ma = df['Close'].rolling(n).mean()
    std = df['Close'].rolling(n).std()
    upper = ma + k * std
    lower = ma - k * std
    return upper, lower

def compute_KDJ(df, n=9, k_period=3, d_period=3):
    low_min = df['Low'].rolling(n).min()
    high_max = df['High'].rolling(n).max()
    rsv = 100 * (df['Close'] - low_min) / (high_max - low_min + 1e-9)
    k = rsv.ewm(com=(k_period - 1), adjust=False).mean()
    d = k.ewm(com=(d_period - 1), adjust=False).mean()
    j = 3 * k - 2 * d
    return k, d, j

# 標籤生成
def generate_labels(df, horizon=5, up_thresh=0.01, down_thresh=-0.01): # 0.1, 0.1
    df['future_return'] = (df['Close'].shift(-horizon) - df['Close']) / df['Close']
    df['label'] = np.select(
        [df['future_return'] > up_thresh, df['future_return'] < down_thresh],
        [1, -1],
        default=0
    )
    df = df.dropna()
    df['label'] = df['label'].map({-1: 0, 0: 1, 1: 2})
    return df

def compute_volume_change(df):
    # 計算成交量日變化率
    df['Volume_Change'] = df['Volume'].pct_change()
    return df

def compute_foreign_investor_net(df):
    # 假設df有 'Foreign_Buy' 與 'Foreign_Sell'欄位
    df['Foreign_Net'] = df['Foreign_Buy'] - df['Foreign_Sell']
    return df

# Attention Layer
class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)
    def build(self, input_shape):
        self.W = self.add_weight(name='att_weight', shape=(input_shape[-1], 1),
                                 initializer='random_normal', trainable=True)
        self.b = self.add_weight(name='att_bias', shape=(input_shape[1], 1),
                                 initializer='zeros', trainable=True)
        super(Attention, self).build(input_shape)
    def call(self, x):
        e = tf.keras.backend.tanh(tf.keras.backend.dot(x, self.W) + self.b)
        a = tf.keras.backend.softmax(e, axis=1)
        output = x * a
        return tf.keras.backend.sum(output, axis=1)

# 原本讀資料
df = yf.download("2330.TW", start="2015-01-01", end="2025-01-01")
df = df.dropna()

# 計算技術指標（示意）
df['MA5'] = df['Close'].rolling(5).mean()
df['MA20'] = df['Close'].rolling(20).mean()
df['RSI14'] = compute_RSI(df['Close'])
df['MACD'], df['MACD_signal'] = compute_MACD(df['Close'])
df['BB_upper'], df['BB_lower'] = compute_BBands(df)
df['K'], df['D'], df['J'] = compute_KDJ(df)

# 加入成交量變化率
df = compute_volume_change(df)

# 加入外資淨買賣（如果你有資料才用）
# df = compute_foreign_investor_net(df)

df = df.dropna()

# 生成標籤
df = generate_labels(df, horizon=5, up_thresh=0.01, down_thresh=-0.01)  # 閥值可調整

# 隨機重採樣，讓三類數量一樣
min_count = df['label'].value_counts().min()
balanced_df = pd.concat([
    df[df['label'] == label].sample(min_count, random_state=42)
    for label in df['label'].unique()
])
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

print("重採樣後類別分布：")
print(balanced_df['label'].value_counts())

# 標準化
features = ['Close', 'MA5', 'MA20', 'RSI14', 'MACD', 'MACD_signal',
            'BB_upper', 'BB_lower', 'K', 'D', 'J']
scaler = StandardScaler()
scaled_features = scaler.fit_transform(balanced_df[features])
joblib.dump(scaler, SCALER_NAME + ".pkl")

# 製作序列
def create_sequences(data, labels, time_steps=20):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i+time_steps])
        y.append(labels[i+time_steps])
    return np.array(X), np.array(y)



X, y = create_sequences(scaled_features, balanced_df['label'].values, time_steps=20)

# 拆訓練測試
split = int(len(X)*0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Multi-head Self Attention
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads)(inputs, inputs)
    x = Dropout(dropout)(x)
    x = LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

    # Feed Forward
    x = Dense(ff_dim, activation="relu")(res)
    x = Dropout(dropout)(x)
    x = Dense(inputs.shape[-1])(x)
    x = LayerNormalization(epsilon=1e-6)(x)
    return x + res

inputs = Input(shape=(X_train.shape[1], X_train.shape[2]))
x = transformer_encoder(inputs, head_size=64, num_heads=4, ff_dim=128, dropout=0.1)
x = GlobalAveragePooling1D()(x)
x = Dropout(0.3)(x)
x = Dense(64, activation='relu')(x)
outputs = Dense(3, activation='softmax')(x)

model = Model(inputs, outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# 早停
early_stop = EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True)

classes = np.unique(y_train)
class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
class_weight_dict = dict(zip(classes, class_weights))

# 訓練
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test),
    class_weight=class_weight_dict,
    callbacks=[early_stop]
)

# 預測與評估
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)
print(classification_report(y_test, y_pred, target_names=['跌', '觀望', '漲']))

model.save(MODEL_NAME)


