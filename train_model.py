import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
import tensorflow as tf
from keras.models import Model
from keras.layers import (LSTM, Dense, Dropout, Input, Bidirectional, BatchNormalization, 
                          MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D, 
                          Conv1D, Concatenate, Add, Activation, GRU, TimeDistributed)
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.optimizers import Adam, AdamW
from keras.regularizers import l1_l2
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import joblib
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTETomek
import warnings
warnings.filterwarnings('ignore')

SCALER_NAME = "scaler/enhanced_model_scaler"
MODEL_NAME = "model/enhanced_model.h5"
fetch_data_start_date = "2005-01-01"
fetch_data_end_date = "2025-01-01"
data_threshold = 0.015
stock_symbol = "2330.TW"  #股票代碼

# 更智能的数据处理
def safe_divide(numerator, denominator):
    """安全除法"""
    if isinstance(numerator, np.ndarray):
        numerator = pd.Series(numerator)
    if isinstance(denominator, np.ndarray):
        denominator = pd.Series(denominator)
    
    # 使用更大的epsilon避免接近零的问题
    result = numerator / (denominator.abs() + 1e-6)
    result = result.replace([np.inf, -np.inf], np.nan)
    return result

def winsorize_data(data, lower_percentile=5, upper_percentile=95):
    """用分位数方法处理异常值"""
    if isinstance(data, np.ndarray):
        data = pd.Series(data)
    
    lower_bound = data.quantile(lower_percentile / 100)
    upper_bound = data.quantile(upper_percentile / 100)
    
    return np.clip(data, lower_bound, upper_bound)

def fetch_market_data(start_date=fetch_data_start_date, end_date=fetch_data_end_date):
    """獲取市場相關數據"""
    print(f"正在下載{stock_symbol}數據...")
    stock_data = yf.download(stock_symbol, start=start_date, end=end_date)
    
    print("正在下載SOX指數數據...")
    sox_data = yf.download("SOXX", start=start_date, end=end_date)
    
    print("正在下載台股加權指數數據...")
    taiwan_index = yf.download("^TWII", start=start_date, end=end_date)
    
    print("正在下載美元指數數據...")
    usd_index = yf.download("DX-Y.NYB", start=start_date, end=end_date)
    
    print("正在下載台幣對美金匯率數據...")
    try:
        usdtwd = yf.download("USDTWD=X", start=start_date, end=end_date)
    except:
        print("警告: 無法獲取USDTWD匯率，使用替代數據")
        usdtwd = yf.download("TWD=X", start=start_date, end=end_date)
    
    # 處理MultiIndex columns
    for data in [stock_data, sox_data, taiwan_index, usd_index, usdtwd]:
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
    
    return stock_data, sox_data, taiwan_index, usd_index, usdtwd

def compute_market_correlation_features(stock_data, sox_data, taiwan_index, usd_index, usdtwd):
    """計算市場相關性特徵"""
    # 對齊所有數據的日期
    common_dates = stock_data.index.intersection(sox_data.index)\
                              .intersection(taiwan_index.index)\
                              .intersection(usd_index.index)\
                              .intersection(usdtwd.index)
    
    # 計算收益率
    stock_returns = stock_data.loc[common_dates, 'Close'].pct_change()
    sox_returns = sox_data.loc[common_dates, 'Close'].pct_change()
    twii_returns = taiwan_index.loc[common_dates, 'Close'].pct_change()
    usd_returns = usd_index.loc[common_dates, 'Close'].pct_change()
    twd_returns = usdtwd.loc[common_dates, 'Close'].pct_change()
    
    market_features = pd.DataFrame(index=common_dates)
    
    # SOX相關特徵
    market_features['SOX_Close'] = sox_data.loc[common_dates, 'Close']
    market_features['SOX_Volume'] = sox_data.loc[common_dates, 'Volume']
    market_features['SOX_Returns_1'] = sox_returns
    market_features['SOX_Returns_5'] = sox_data.loc[common_dates, 'Close'].pct_change(5)
    market_features['SOX_MA20'] = market_features['SOX_Close'].rolling(20).mean()
    market_features['SOX_RSI'] = compute_RSI(market_features['SOX_Close'])
    
    # 台股加權指數特徵
    market_features['TWII_Close'] = taiwan_index.loc[common_dates, 'Close']
    market_features['TWII_Volume'] = taiwan_index.loc[common_dates, 'Volume']
    market_features['TWII_Returns_1'] = twii_returns
    market_features['TWII_Returns_5'] = taiwan_index.loc[common_dates, 'Close'].pct_change(5)
    market_features['TWII_MA20'] = market_features['TWII_Close'].rolling(20).mean()
    market_features['TWII_RSI'] = compute_RSI(market_features['TWII_Close'])
    
    # 美元指數特徵
    market_features['USD_Close'] = usd_index.loc[common_dates, 'Close']
    market_features['USD_Returns_1'] = usd_returns
    market_features['USD_MA10'] = market_features['USD_Close'].rolling(10).mean()
    
    # 台幣匯率特徵
    market_features['USDTWD_Close'] = usdtwd.loc[common_dates, 'Close']
    market_features['USDTWD_Returns_1'] = twd_returns
    market_features['USDTWD_MA10'] = market_features['USDTWD_Close'].rolling(10).mean()
    
    # 相關性特徵 (滾動相關性)
    correlation_window = 20
    market_features['stock_SOX_Corr'] = stock_returns.rolling(correlation_window).corr(sox_returns)
    market_features['stock_TWII_Corr'] = stock_returns.rolling(correlation_window).corr(twii_returns)
    market_features['stock_USD_Corr'] = stock_returns.rolling(correlation_window).corr(usd_returns)
    market_features['stock_TW_Corr'] = stock_returns.rolling(correlation_window).corr(twd_returns)
    
    # 相對強度特徵
    market_features['stock_SOX_Ratio'] = safe_divide(stock_data.loc[common_dates, 'Close'], 
                                                     sox_data.loc[common_dates, 'Close'])
    market_features['stock_TWII_Ratio'] = safe_divide(stock_data.loc[common_dates, 'Close'], 
                                                      taiwan_index.loc[common_dates, 'Close'])
    
    # 相對表現 (outperformance)
    market_features['stock_vs_SOX'] = stock_returns - sox_returns
    market_features['stock_vs_TWII'] = stock_returns - twii_returns
    
    # 市場情緒指標
    market_features['SOX_Momentum'] = market_features['SOX_Close'] / market_features['SOX_MA20'] - 1
    market_features['TWII_Momentum'] = market_features['TWII_Close'] / market_features['TWII_MA20'] - 1
    
    # 波動率比較
    vol_window = 20
    market_features['SOX_Volatility'] = sox_returns.rolling(vol_window).std()
    market_features['TWII_Volatility'] = twii_returns.rolling(vol_window).std()
    market_features['USD_Volatility'] = usd_returns.rolling(vol_window).std()
    
    return market_features

def compute_advanced_technical_indicators(df):
    """計算更多高質量技術指標"""
    data = df.copy()
    
    # 基礎價格特徵
    data['Returns_1'] = data['Close'].pct_change(1)
    data['Returns_3'] = data['Close'].pct_change(3)
    data['Returns_5'] = data['Close'].pct_change(5)
    data['Returns_10'] = data['Close'].pct_change(10)
    
    # 多時間框架移動平均
    periods = [5, 10, 20, 50, 100]
    for period in periods:
        data[f'MA{period}'] = data['Close'].rolling(period).mean()
        data[f'EMA{period}'] = data['Close'].ewm(span=period).mean()
        data[f'MA{period}_slope'] = data[f'MA{period}'].diff(5)  # 趨勢斜率
        data[f'Price_MA{period}_ratio'] = safe_divide(data['Close'], data[f'MA{period}'])
        data[f'Price_EMA{period}_ratio'] = safe_divide(data['Close'], data[f'EMA{period}'])
        
    # MA交叉信號
    data['MA5_MA20_cross'] = np.where(data['MA5'] > data['MA20'], 1, -1)
    data['MA10_MA50_cross'] = np.where(data['MA10'] > data['MA50'], 1, -1)
    data['MA20_MA100_cross'] = np.where(data['MA20'] > data['MA100'], 1, -1)
    
    # 改進的RSI (多時間框架)
    for period in [7, 14, 21, 28]:
        data[f'RSI{period}'] = compute_RSI(data['Close'], period)
        data[f'RSI{period}_signal'] = np.where(data[f'RSI{period}'] > 70, -1, 
                                              np.where(data[f'RSI{period}'] < 30, 1, 0))
        # RSI背離檢測
        data[f'RSI{period}_divergence'] = detect_divergence(data['Close'], data[f'RSI{period}'])
    
    # MACD系統 (多參數)
    for fast, slow, signal in [(12, 26, 9), (5, 35, 5), (19, 39, 9)]:
        macd_line, signal_line, histogram = compute_MACD(data['Close'], fast, slow, signal)
        suffix = f"_{fast}_{slow}_{signal}"
        data[f'MACD{suffix}'] = macd_line
        data[f'MACD_signal{suffix}'] = signal_line
        data[f'MACD_histogram{suffix}'] = histogram
        data[f'MACD_cross{suffix}'] = np.where(macd_line > signal_line, 1, -1)
    
    # 布林帶系統 (多參數)
    for period, std_dev in [(20, 2), (20, 2.5), (10, 1.5)]:
        bb_upper, bb_lower, bb_middle = compute_bollinger_bands(data['Close'], period, std_dev)
        suffix = f"_{period}_{std_dev}"
        data[f'BB_upper{suffix}'] = bb_upper
        data[f'BB_lower{suffix}'] = bb_lower
        data[f'BB_middle{suffix}'] = bb_middle
        data[f'BB_width{suffix}'] = safe_divide((bb_upper - bb_lower), bb_middle)
        data[f'BB_position{suffix}'] = safe_divide((data['Close'] - bb_lower), (bb_upper - bb_lower))
        data[f'BB_squeeze{suffix}'] = (data[f'BB_width{suffix}'] < data[f'BB_width{suffix}'].rolling(20).mean() * 0.8).astype(int)
    
    # KDJ優化 (多參數)
    for n, k_period, d_period in [(9, 3, 3), (14, 5, 5), (21, 7, 7)]:
        k, d, j = compute_KDJ(data, n, k_period, d_period)
        suffix = f"_{n}_{k_period}_{d_period}"
        data[f'K{suffix}'] = k
        data[f'D{suffix}'] = d
        data[f'J{suffix}'] = j
        data[f'KD_cross{suffix}'] = np.where(k > d, 1, -1)
    
    # 其他技術指標
    data['Williams_R'] = compute_williams_r(data)
    data['CCI'] = compute_cci(data)
    data['Stoch_RSI'] = compute_stoch_rsi(data)
    data['ADX'] = compute_adx(data)
    data['PSAR'] = compute_parabolic_sar(data)
    
    # 成交量分析 (進階)
    data['Volume_MA5'] = data['Volume'].rolling(5).mean()
    data['Volume_MA20'] = data['Volume'].rolling(20).mean()
    data['Volume_MA50'] = data['Volume'].rolling(50).mean()
    data['Volume_ratio'] = safe_divide(data['Volume'], data['Volume_MA20'])
    data['Volume_trend'] = data['Volume_MA5'] / data['Volume_MA20'] - 1
    
    # OBV (On Balance Volume)
    data['OBV'] = compute_obv(data)
    data['OBV_MA'] = data['OBV'].rolling(20).mean()
    data['OBV_signal'] = np.where(data['OBV'] > data['OBV_MA'], 1, -1)
    
    # 價量背離檢測
    data['Price_Volume_correlation'] = data['Close'].rolling(10).corr(data['Volume']).fillna(0)
    data['Volume_price_trend'] = np.where(
        (data['Returns_5'] > 0) & (data['Volume_ratio'] > 1.2), 1,  # 放量上漲
        np.where((data['Returns_5'] < 0) & (data['Volume_ratio'] > 1.2), -1, 0)  # 放量下跌
    )
    
    # 波動率特徵 (擴展)
    for period in [7, 14, 21, 30]:
        data[f'ATR{period}'] = compute_ATR(data, period)
        data[f'ATR{period}_ratio'] = safe_divide(data[f'ATR{period}'], data['Close'])
        data[f'Volatility{period}'] = data['Returns_1'].rolling(period).std()
    
    # 高低點分析
    data['High_Low_ratio'] = safe_divide((data['High'] - data['Low']), data['Close'])
    data['Close_position'] = safe_divide((data['Close'] - data['Low']), (data['High'] - data['Low']))
    data['Upper_shadow'] = safe_divide((data['High'] - np.maximum(data['Open'], data['Close'])), data['Close'])
    data['Lower_shadow'] = safe_divide((np.minimum(data['Open'], data['Close']) - data['Low']), data['Close'])
    
    # 趨勢強度和方向
    data['Trend_strength'] = compute_trend_strength(data['Close'])
    data['Price_velocity'] = data['Close'].diff() / data['Close'].shift()
    data['Price_acceleration'] = data['Price_velocity'].diff()
    
    # 支撐阻力位 (多時間框架)
    for window in [10, 20, 50]:
        data[f'Resistance_distance_{window}'] = compute_resistance_distance(data, window)
        data[f'Support_distance_{window}'] = compute_support_distance(data, window)
    
    # 蠟燭圖形態識別
    data['Doji'] = ((data['Close'] - data['Open']).abs() / (data['High'] - data['Low']) < 0.1).astype(int)
    data['Hammer'] = identify_hammer(data)
    data['Engulfing'] = identify_engulfing_pattern(data)
    
    # 數據清理
    for col in data.select_dtypes(include=[np.number]).columns:
        if col not in ['Open', 'High', 'Low', 'Close', 'Volume']:
            data[col] = winsorize_data(data[col])
            data[col] = data[col].fillna(method='ffill').fillna(0)
    
    return data

# 新增技術指標函數
def compute_RSI(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = safe_divide(avg_gain, avg_loss)
    rsi = 100 - safe_divide(100, (1 + rs))
    return rsi.fillna(50)

def compute_MACD(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast).mean()
    ema_slow = series.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

def compute_bollinger_bands(series, period=20, std_dev=2):
    ma = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = ma + (std * std_dev)
    lower = ma - (std * std_dev)
    return upper, lower, ma

def compute_KDJ(df, n=9, k_period=3, d_period=3):
    low_min = df['Low'].rolling(n).min()
    high_max = df['High'].rolling(n).max()
    rsv = 100 * safe_divide((df['Close'] - low_min), (high_max - low_min))
    k = rsv.ewm(com=(k_period - 1), adjust=False).mean()
    d = k.ewm(com=(d_period - 1), adjust=False).mean()
    j = 3 * k - 2 * d
    return k, d, j

def compute_williams_r(df, period=14):
    high_max = df['High'].rolling(period).max()
    low_min = df['Low'].rolling(period).min()
    wr = -100 * safe_divide((high_max - df['Close']), (high_max - low_min))
    return wr

def compute_cci(df, period=20):
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    ma = typical_price.rolling(period).mean()
    mad = typical_price.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())))
    cci = safe_divide((typical_price - ma), (0.015 * mad))
    return cci

def compute_stoch_rsi(df, period=14):
    rsi = compute_RSI(df['Close'], period)
    stoch_rsi = safe_divide((rsi - rsi.rolling(period).min()), 
                           (rsi.rolling(period).max() - rsi.rolling(period).min())) * 100
    return stoch_rsi

def compute_adx(df, period=14):
    plus_dm = df['High'].diff()
    minus_dm = df['Low'].diff() * -1
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    
    tr = compute_ATR(df, 1) * df.shape[0]  # True Range
    plus_di = 100 * (plus_dm.rolling(period).mean() / tr.rolling(period).mean())
    minus_di = 100 * (minus_dm.rolling(period).mean() / tr.rolling(period).mean())
    
    dx = 100 * safe_divide((plus_di - minus_di).abs(), (plus_di + minus_di))
    adx = dx.rolling(period).mean()
    return adx

def compute_parabolic_sar(df, acceleration=0.02, maximum=0.2):
    # 簡化版本的Parabolic SAR
    high, low, close = df['High'], df['Low'], df['Close']
    sar = close.copy()
    
    # 這是一個簡化的實現，實際的PSAR算法更複雜
    for i in range(1, len(df)):
        if close.iloc[i] > close.iloc[i-1]:  # 上升趋势
            sar.iloc[i] = min(low.iloc[i-1], sar.iloc[i-1])
        else:  # 下降趋势
            sar.iloc[i] = max(high.iloc[i-1], sar.iloc[i-1])
    
    return sar

def compute_obv(df):
    obv = [0]
    for i in range(1, len(df)):
        if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
            obv.append(obv[-1] + df['Volume'].iloc[i])
        elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
            obv.append(obv[-1] - df['Volume'].iloc[i])
        else:
            obv.append(obv[-1])
    return pd.Series(obv, index=df.index)

def compute_ATR(df, period=14):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    return true_range.rolling(period).mean()

def compute_trend_strength(series, period=20):
    """計算趨勢強度"""
    returns = series.pct_change()
    positive_returns = returns.where(returns > 0, 0)
    negative_returns = returns.where(returns < 0, 0).abs()
    
    avg_positive = positive_returns.rolling(period).mean()
    avg_negative = negative_returns.rolling(period).mean()
    
    trend_strength = safe_divide(avg_positive, (avg_positive + avg_negative))
    return trend_strength.fillna(0.5)

def compute_resistance_distance(df, window=20):
    """計算到阻力位的距離"""
    resistance = df['High'].rolling(window).max()
    distance = safe_divide((resistance - df['Close']), df['Close'])
    return distance

def compute_support_distance(df, window=20):
    """計算到支撐位的距離"""
    support = df['Low'].rolling(window).min()
    distance = safe_divide((df['Close'] - support), df['Close'])
    return distance

def detect_divergence(price, indicator, window=20):
    """檢測價格與指標的背離"""
    price_trend = price.rolling(window).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
    indicator_trend = indicator.rolling(window).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
    
    # 正背離：價格下跌但指標上升
    bullish_divergence = (price_trend < 0) & (indicator_trend > 0)
    # 負背離：價格上漲但指標下跌
    bearish_divergence = (price_trend > 0) & (indicator_trend < 0)
    
    return np.where(bullish_divergence, 1, np.where(bearish_divergence, -1, 0))

def identify_hammer(df):
    """識別錘子線形態"""
    body = (df['Close'] - df['Open']).abs()
    upper_shadow = df['High'] - df[['Close', 'Open']].max(axis=1)
    lower_shadow = df[['Close', 'Open']].min(axis=1) - df['Low']
    
    hammer = (lower_shadow > 2 * body) & (upper_shadow < 0.1 * body)
    return hammer.astype(int)

def identify_engulfing_pattern(df):
    """識別包吞形態"""
    prev_body = (df['Close'].shift() - df['Open'].shift()).abs()
    curr_body = (df['Close'] - df['Open']).abs()
    
    bullish_engulfing = (df['Close'].shift() < df['Open'].shift()) & \
                       (df['Close'] > df['Open']) & \
                       (curr_body > prev_body) & \
                       (df['Open'] < df['Close'].shift()) & \
                       (df['Close'] > df['Open'].shift())
    
    bearish_engulfing = (df['Close'].shift() > df['Open'].shift()) & \
                       (df['Close'] < df['Open']) & \
                       (curr_body > prev_body) & \
                       (df['Open'] > df['Close'].shift()) & \
                       (df['Close'] < df['Open'].shift())
    
    return np.where(bullish_engulfing, 1, np.where(bearish_engulfing, -1, 0))

def generate_improved_labels(df, horizon=2, method='adaptive', threshold=data_threshold):
    """改進的標籤生成方法"""
    data = df.copy()
    
    # 計算未來收益率
    future_return = (data['Close'].shift(-horizon) - data['Close']) / data['Close']
    
    if method == 'adaptive':
        # 自適應閾值：基於滾動波動率
        window = 60
        volatility = data['Close'].pct_change().rolling(window).std()
        upper_threshold = volatility * 1.5  # 動態閾值
        lower_threshold = -volatility * 1.5
    elif method == 'quantile':
        # 動態閾值：基於滾動分位數
        window = 60
        upper_threshold = future_return.rolling(window).quantile(0.75)
        lower_threshold = future_return.rolling(window).quantile(0.25)
    else:
        # 固定閾值
        upper_threshold = threshold
        lower_threshold = -threshold
    
    # 生成標籤
    conditions = [
        future_return > upper_threshold,
        future_return < lower_threshold
    ]
    choices = [2, 0]  # 漲、跌
    data['label'] = np.select(conditions, choices, default=1)
    
    # 過濾掉邊界樣本
    data['future_return'] = future_return
    data = data.dropna()
    
    return data

def create_enhanced_transformer_block(inputs, d_model=64, num_heads=4, dropout_rate=0.2):
    """創建增強的Transformer塊 - 修復版本"""
    
    # 獲取輸入維度並確保一致性
    input_dim = int(inputs.shape[-1])
    
    # 使用統一的維度，避免不匹配問題
    # 如果輸入維度不能被num_heads整除，調整d_model
    if input_dim % num_heads != 0:
        d_model = ((input_dim // num_heads) + 1) * num_heads
        # 使用Dense層調整輸入維度到d_model
        adjusted_inputs = Dense(d_model, activation='linear', name=f'dim_adjust_{np.random.randint(10000)}')(inputs)
    else:
        d_model = input_dim
        adjusted_inputs = inputs
    
    # Multi-head attention
    attention_output = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=d_model // num_heads,
        dropout=dropout_rate,
        name=f'mha_{np.random.randint(10000)}'
    )(adjusted_inputs, adjusted_inputs)
    
    attention_output = Dropout(dropout_rate)(attention_output)
    attention_output = LayerNormalization(epsilon=1e-6)(Add()([adjusted_inputs, attention_output]))
    
    # Feed forward network - 確保維度匹配
    ffn_layer1 = Dense(d_model * 2, activation='relu')(attention_output)
    ffn_layer1 = Dropout(dropout_rate)(ffn_layer1)
    ffn_output = Dense(d_model)(ffn_layer1)  # 輸出維度與attention_output相同
    ffn_output = Dropout(dropout_rate)(ffn_output)
    
    # 最終的residual connection和layer normalization
    final_output = LayerNormalization(epsilon=1e-6)(Add()([attention_output, ffn_output]))
    
    return final_output

def create_ultimate_model(input_shape, num_classes=3):
    """創建終極增強模型架構 - 修復版本"""
    inputs = Input(shape=input_shape)
    
    # 1. 多尺度CNN特徵提取層
    cnn_outputs = []
    filters = [64, 64, 32]
    
    for i, kernel_size in enumerate([3, 5, 7, 11]):
        cnn = Conv1D(filters[min(i, len(filters)-1)], kernel_size, 
                     padding='same', activation='relu')(inputs)
        cnn = BatchNormalization()(cnn)
        cnn = Dropout(0.2)(cnn)
        cnn = Conv1D(filters[min(i, len(filters)-1)], kernel_size, 
                     padding='same', activation='relu')(cnn)
        cnn = BatchNormalization()(cnn)
        cnn_outputs.append(cnn)
    
    cnn_merged = Concatenate()(cnn_outputs)
    cnn_merged = Dropout(0.3)(cnn_merged)
    
    # 2. 混合RNN層 (LSTM + GRU)
    lstm_out = Bidirectional(LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.1))(inputs)
    lstm_out = BatchNormalization()(lstm_out)
    lstm_out = Bidirectional(LSTM(32, return_sequences=True, dropout=0.2, recurrent_dropout=0.1))(lstm_out)
    lstm_out = BatchNormalization()(lstm_out)
    
    gru_out = Bidirectional(GRU(48, return_sequences=True, dropout=0.2, recurrent_dropout=0.1))(inputs)
    gru_out = BatchNormalization()(gru_out)
    gru_out = Bidirectional(GRU(24, return_sequences=True, dropout=0.2, recurrent_dropout=0.1))(gru_out)
    gru_out = BatchNormalization()(gru_out)
    
    # 3. 簡化的Transformer編碼器 - 統一維度處理
    # 先將輸入調整到統一維度
    transformer_input = Dense(64, activation='linear')(inputs)
    
    transformer_out = transformer_input
    for i in range(2):  # 使用2層Transformer
        transformer_out = create_enhanced_transformer_block(
            transformer_out, 
            d_model=64,  # 使用固定的64維度
            num_heads=4, 
            dropout_rate=0.2
        )
    
    # 4. 特徵融合 - 統一所有分支的維度
    # 調整所有分支到相同的特徵維度
    feature_dim = 48
    
    cnn_adjusted = Dense(feature_dim)(cnn_merged)
    lstm_adjusted = Dense(feature_dim)(lstm_out)
    gru_adjusted = Dense(feature_dim)(gru_out)
    transformer_adjusted = Dense(feature_dim)(transformer_out)
    
    # 5. 注意力權重計算
    from keras.layers import Lambda, GlobalAveragePooling1D, GlobalMaxPooling1D
    
    cnn_pooled = GlobalAveragePooling1D()(cnn_adjusted)
    lstm_pooled = GlobalAveragePooling1D()(lstm_adjusted)
    gru_pooled = GlobalAveragePooling1D()(gru_adjusted)
    transformer_pooled = GlobalAveragePooling1D()(transformer_adjusted)
    
    # 計算注意力權重
    cnn_importance = Dense(1, activation='sigmoid')(cnn_pooled)
    lstm_importance = Dense(1, activation='sigmoid')(lstm_pooled)
    gru_importance = Dense(1, activation='sigmoid')(gru_pooled)
    transformer_importance = Dense(1, activation='sigmoid')(transformer_pooled)
    
    # 應用注意力權重
    def apply_attention_weight(inputs):
        features, weight = inputs
        # 將權重擴展到匹配特徵的維度 [batch, time, features] * [batch, 1, 1]
        weight_expanded = tf.expand_dims(weight, axis=1)  # [batch, 1, 1]
        return features * weight_expanded
    
    cnn_weighted = Lambda(apply_attention_weight)([cnn_adjusted, cnn_importance])
    lstm_weighted = Lambda(apply_attention_weight)([lstm_adjusted, lstm_importance])
    gru_weighted = Lambda(apply_attention_weight)([gru_adjusted, gru_importance])
    transformer_weighted = Lambda(apply_attention_weight)([transformer_adjusted, transformer_importance])
    
    # 6. 最終特徵融合
    merged_features = Concatenate()([cnn_weighted, lstm_weighted, gru_weighted, transformer_weighted])
    
    # 7. 最終的注意力層 - 調整參數以適應新的特徵維度
    final_feature_dim = feature_dim * 4  # 192
    
    # 確保能被head數整除
    adjusted_final_dim = 192  # 192 / 8 = 24, 可以被8整除
    if final_feature_dim != adjusted_final_dim:
        merged_features = Dense(adjusted_final_dim)(merged_features)
    
    final_attention = MultiHeadAttention(
        num_heads=8,
        key_dim=24,  # 192 / 8
        dropout=0.3
    )(merged_features, merged_features)
    
    final_attention = LayerNormalization(epsilon=1e-6)(Add()([merged_features, final_attention]))
    
    # 8. 全局池化
    global_avg_pool = GlobalAveragePooling1D()(final_attention)
    global_max_pool = GlobalMaxPooling1D()(final_attention)
    
    # 9. 深度分類網絡
    combined_features = Concatenate()([global_avg_pool, global_max_pool])
    
    # 分類層 - 更保守的架構以避免過擬合
    x = Dense(256, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(combined_features)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    
    x = Dense(128, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    # 輸出層
    outputs = Dense(num_classes, activation='softmax', name='predictions')(x)
    
    model = Model(inputs, outputs)
    return model

def create_sequences_with_advanced_overlap(data, labels, time_steps=30, overlap=0.6):
    """創建高級重疊序列"""
    step_size = max(1, int(time_steps * (1 - overlap)))
    X, y = [], []
    
    for i in range(0, len(data) - time_steps, step_size):
        sequence = data[i:i+time_steps]
        label = labels[i+time_steps]
        
        # 添加序列變異以增加多樣性
        X.append(sequence)
        y.append(label)
        
        # 添加輕微噪聲版本
        if np.random.random() > 0.7:
            noisy_sequence = sequence + np.random.normal(0, 0.005, sequence.shape)
            X.append(noisy_sequence)
            y.append(label)
    
    return np.array(X), np.array(y)

def ultimate_data_augmentation(X, y, augmentation_factor=2):
    """終極數據增強技術"""
    X_aug, y_aug = [], []
    
    for i in range(len(X)):
        # 原始數據
        X_aug.append(X[i])
        y_aug.append(y[i])
        
        for _ in range(augmentation_factor):
            augmented_x = X[i].copy()
            
            # 1. 高斯噪聲 (adaptive)
            noise_scale = np.std(augmented_x) * 0.02
            noise = np.random.normal(0, noise_scale, augmented_x.shape)
            augmented_x += noise
            
            # 2. 特徵dropout
            if np.random.random() > 0.8:
                dropout_mask = np.random.random(augmented_x.shape[1]) > 0.1
                augmented_x[:, ~dropout_mask] = 0
            
            # 3. 時間彎曲 (time warping)
            if np.random.random() > 0.85:
                warp_factor = np.random.uniform(0.9, 1.1)
                new_length = int(augmented_x.shape[0] * warp_factor)
                if new_length > 0:
                    indices = np.linspace(0, augmented_x.shape[0]-1, new_length).astype(int)
                    warped = augmented_x[indices]
                    if warped.shape[0] == augmented_x.shape[0]:
                        augmented_x = warped
            
            # 4. 幅度縮放
            if np.random.random() > 0.9:
                scale_factor = np.random.uniform(0.95, 1.05)
                augmented_x *= scale_factor
            
            # 5. 特徵混合
            if np.random.random() > 0.85:
                mix_ratio = np.random.uniform(0.1, 0.3)
                other_idx = np.random.randint(0, len(X))
                if y[other_idx] == y[i]:  # 只與同類別混合
                    augmented_x = augmented_x * (1 - mix_ratio) + X[other_idx] * mix_ratio
            
            X_aug.append(augmented_x)
            y_aug.append(y[i])
    
    return np.array(X_aug), np.array(y_aug)

# 主程序
def main():
    print("=== 終極增強股票預測模型 ===")
    print("正在下載市場數據...")
    
    # 獲取所有市場數據
    stock_data, sox_data, taiwan_index, usd_index, usdtwd = fetch_market_data()
    
    # 處理台積電數據
    if isinstance(stock_data.columns, pd.MultiIndex):
        stock_data.columns = stock_data.columns.get_level_values(0)
    
    stock_data = stock_data.dropna()
    print(f"{stock_symbol}原始數據形狀: {stock_data.shape}")
    
    # 計算台積電技術指標
    print(f"正在計算{stock_symbol}技術指標...")
    stock_data = compute_advanced_technical_indicators(stock_data)
    print(f"{stock_symbol}技術指標完成，數據形狀: {stock_data.shape}")
    
    # 計算市場相關特徵
    print("正在計算市場相關特徵...")
    market_features = compute_market_correlation_features(
        stock_data, sox_data, taiwan_index, usd_index, usdtwd
    )
    print(f"市場特徵計算完成，數據形狀: {market_features.shape}")
    
    # 合併所有特徵
    print("正在合併所有特徵...")
    common_dates = stock_data.index.intersection(market_features.index)
    
    # 對齊數據
    stock_aligned = stock_data.loc[common_dates]
    market_aligned = market_features.loc[common_dates]
    
    # 合併數據
    combined_data = pd.concat([stock_aligned, market_aligned], axis=1)
    combined_data = combined_data.dropna()
    print(f"合併後數據形狀: {combined_data.shape}")
    
    # 生成改進標籤
    print("正在生成標籤...")
    combined_data = generate_improved_labels(combined_data, horizon=2, method='adaptive')
    print("標籤分布：")
    print(combined_data['label'].value_counts())
    print(f"標籤比例: {combined_data['label'].value_counts(normalize=True)}")
    
    # 選擇最重要的特徵
    important_features = [
        # 價格特徵
        'Returns_1', 'Returns_3', 'Returns_5', 'Returns_10',
        
        # 移動平均系統
        'MA5', 'MA10', 'MA20', 'MA50', 'MA100',
        'EMA5', 'EMA10', 'EMA20', 'EMA50',
        'Price_MA5_ratio', 'Price_MA20_ratio', 'Price_MA50_ratio',
        'Price_EMA10_ratio', 'Price_EMA20_ratio',
        'MA5_slope', 'MA10_slope', 'MA20_slope',
        'MA5_MA20_cross', 'MA10_MA50_cross', 'MA20_MA100_cross',
        
        # 多參數技術指標
        'RSI7', 'RSI14', 'RSI21', 'RSI28',
        'RSI14_signal', 'RSI14_divergence',
        'MACD_12_26_9', 'MACD_signal_12_26_9', 'MACD_histogram_12_26_9', 'MACD_cross_12_26_9',
        'MACD_5_35_5', 'MACD_19_39_9',
        
        # 布林帶系統
        'BB_width_20_2', 'BB_position_20_2', 'BB_squeeze_20_2',
        'BB_width_20_2.5', 'BB_width_10_1.5',
        
        # KDJ系統
        'K_9_3_3', 'D_9_3_3', 'J_9_3_3', 'KD_cross_9_3_3',
        'K_14_5_5', 'D_14_5_5',
        
        # 其他技術指標
        'Williams_R', 'CCI', 'Stoch_RSI', 'ADX',
        
        # 成交量分析
        'Volume_ratio', 'Volume_trend', 'Volume_price_trend', 
        'Price_Volume_correlation', 'OBV_signal',
        
        # 波動率特徵
        'ATR7_ratio', 'ATR14_ratio', 'ATR21_ratio', 'ATR30_ratio',
        'Volatility7', 'Volatility14', 'Volatility21', 'Volatility30',
        
        # 位置和形態
        'High_Low_ratio', 'Close_position', 'Upper_shadow', 'Lower_shadow',
        'Doji', 'Hammer', 'Engulfing',
        
        # 趨勢分析
        'Trend_strength', 'Price_velocity', 'Price_acceleration',
        'Resistance_distance_10', 'Resistance_distance_20', 'Resistance_distance_50',
        'Support_distance_10', 'Support_distance_20', 'Support_distance_50',
        
        # SOX相關特徵
        'SOX_Returns_1', 'SOX_Returns_5', 'SOX_RSI', 'SOX_Momentum',
        'SOX_Volatility', 'stock_SOX_Corr', 'stock_SOX_Ratio', 'stock_vs_SOX',
        
        # 台股指數特徵
        'TWII_Returns_1', 'TWII_Returns_5', 'TWII_RSI', 'TWII_Momentum',
        'TWII_Volatility', 'stock_TWII_Corr', 'stock_TWII_Ratio', 'stock_vs_TWII',
        
        # 匯率相關特徵
        'USD_Returns_1', 'USD_Volatility', 'stock_USD_Corr',
        'USDTWD_Returns_1', 'stock_TW_Corr'
    ]
    
    # 確保特徵存在並過濾
    existing_features = [f for f in important_features if f in combined_data.columns]
    missing_features = [f for f in important_features if f not in combined_data.columns]
    
    print(f"使用特徵數量: {len(existing_features)}")
    print(f"缺少特徵數量: {len(missing_features)}")
    if missing_features:
        print(f"缺少的特徵: {missing_features[:10]}...")  # 只顯示前10個
    
    # 特徵數據準備
    feature_data = combined_data[existing_features].copy()
    
    # 最終數據清理
    for col in feature_data.columns:
        feature_data[col] = feature_data[col].replace([np.inf, -np.inf], np.nan)
        feature_data[col] = feature_data[col].fillna(method='ffill').fillna(0)
    
    print(f"最終特徵數據形狀: {feature_data.shape}")
    
    # 使用RobustScaler提高對異常值的抗性
    print("正在標準化特徵...")
    scaler = RobustScaler()
    global scaled_features
    scaled_features = scaler.fit_transform(feature_data)
    joblib.dump(scaler, SCALER_NAME + ".pkl")
    
    # 創建高級序列
    print("正在創建高級序列...")
    labels = combined_data.loc[feature_data.index, 'label'].values
    X, y = create_sequences_with_advanced_overlap(
        scaled_features, labels, time_steps=30, overlap=0.6
    )
    print(f"序列形狀: X={X.shape}, y={y.shape}")
    
    # 時間序列分割
    test_size = 0.15  # 減少測試集比例以增加訓練數據
    train_size = int(len(X) * (1 - test_size))
    
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    print(f"訓練集: {X_train.shape}, 測試集: {X_test.shape}")
    
    # 終極數據增強
    print("正在進行終極數據增強...")
    X_train_aug, y_train_aug = ultimate_data_augmentation(X_train, y_train, augmentation_factor=2)
    print(f"增強後數據: {X_train_aug.shape}")
    
    # 高級重採樣
    print("正在處理類別不平衡...")
    X_train_reshaped = X_train_aug.reshape(X_train_aug.shape[0], -1)
    
    # 使用ADASYN進行自適應採樣
    adasyn = ADASYN(random_state=42, n_neighbors=3)
    try:
        X_train_balanced, y_train_balanced = adasyn.fit_resample(X_train_reshaped, y_train_aug)
        X_train_balanced = X_train_balanced.reshape(-1, X_train_aug.shape[1], X_train_aug.shape[2])
        print("使用ADASYN重採樣成功")
    except:
        print("ADASYN失敗，使用SMOTE")
        smote = SMOTE(random_state=42, k_neighbors=3)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_reshaped, y_train_aug)
        X_train_balanced = X_train_balanced.reshape(-1, X_train_aug.shape[1], X_train_aug.shape[2])
    
    print("最終訓練數據標籤分布:")
    unique, counts = np.unique(y_train_balanced, return_counts=True)
    for label, count in zip(unique, counts):
        label_name = ['跌', '觀望', '漲'][int(label)]
        print(f"{label_name}: {count} ({count/len(y_train_balanced)*100:.1f}%)")
    
    # 創建終極模型
    print("正在創建終極增強模型...")
    model = create_ultimate_model(
        input_shape=(X_train_balanced.shape[1], X_train_balanced.shape[2]),
        num_classes=3
    )
    
    # 使用AdamW優化器 (更好的正則化)
    optimizer = AdamW(
        learning_rate=0.001,
        weight_decay=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07
    )
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"模型參數數量: {model.count_params():,}")
    
    # 高級回調函數
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=30,
            restore_best_weights=True,
            verbose=1,
            min_delta=0.0005
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=10,
            min_lr=1e-7,
            verbose=1,
            cooldown=5
        ),
        ModelCheckpoint(
            MODEL_NAME,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1,
            save_weights_only=False
        )
    ]
    
    # 計算精細化類別權重
    classes = np.unique(y_train_balanced)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train_balanced)
    class_weight_dict = dict(zip(classes, class_weights))
    
    # 手動調整權重
    class_weight_dict[0] *= 7  # 跌 - 提高重要性
    class_weight_dict[2] *= 1.4  # 漲 - 最重要
    class_weight_dict[1] *= 0.6  # 觀望 - 降低重要性
    
    print(f"調整後類別權重: {class_weight_dict}")
    
    # 訓練終極模型
    print("開始訓練終極增強模型...")
    history = model.fit(
        X_train_balanced, y_train_balanced,
        epochs=300,
        batch_size=64,  # 增加批次大小提高穩定性
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1
    )
    
    # 最終評估
    print("\n=== 終極模型評估結果 ===")

    print("\n驗證資料：")
    model.load_weights(MODEL_NAME)
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)
    
    print("詳細分類報告:")
    print(classification_report(y_test, y_pred, target_names=['跌', '觀望', '漲']))

if __name__ == "__main__":
    main()