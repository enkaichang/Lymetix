import yfinance as yf
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import load_model
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 模型和標準化器文件名（需要與訓練時保持一致）
SCALER_NAME = "E:\Stock_114_8_9\scaler\enhanced_model_scaler.pkl"
MODEL_NAME = "E:\Stock_114_8_9\model\enhanced_model.h5"

class stockStockPredictor:
    def __init__(self, scaler_path=None, model_path=None):
        """初始化預測器"""
        self.scaler_path = scaler_path or SCALER_NAME
        self.model_path = model_path or MODEL_NAME
        self.scaler = None
        self.model = None
        self.feature_names = self._get_feature_names()
        
    def _get_feature_names(self):
        """獲取特徵名稱列表（需要與訓練時保持一致）"""
        return [
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
    
    def load_model_and_scaler(self):
        """載入預訓練模型和標準化器"""
        try:
            print("正在載入模型和標準化器...")
            self.scaler = joblib.load(self.scaler_path)
            
            # 定義自定義函數以解決載入問題
            def apply_attention_weight(inputs):
                features, weight = inputs
                weight_expanded = tf.expand_dims(weight, axis=1)
                return features * weight_expanded
            
            # 使用custom_objects載入模型
            custom_objects = {
                'apply_attention_weight': apply_attention_weight
            }
            
            try:
                # 方法1: 使用custom_objects
                self.model = load_model(self.model_path, custom_objects=custom_objects)
                print("使用custom_objects成功載入模型！")
            except Exception as e1:
                try:
                    # 方法2: 只載入權重
                    print("嘗試替代載入方法...")
                    # 需要重新創建模型架構然後載入權重
                    self.model = self._load_model_weights_only()
                    print("使用權重載入方法成功！")
                except Exception as e2:
                    print(f"所有載入方法都失敗:")
                    print(f"方法1錯誤: {str(e1)}")
                    print(f"方法2錯誤: {str(e2)}")
                    return False
            
            print("模型和標準化器載入成功！")
            return True
        except Exception as e:
            print(f"載入模型失敗: {str(e)}")
            print("\n可能的解決方案:")
            print("1. 重新訓練模型並保存")
            print("2. 使用model.save_weights()保存權重而不是完整模型")
            print("3. 確認模型文件完整性")
            return False
    
    def _load_model_weights_only(self):
        """只載入模型權重的方法（需要重建架構）"""
        print("正在重建模型架構...")
        
        # 這裡需要重新創建與訓練時相同的模型架構
        # 由於原始架構比較複雜，我們創建一個簡化版本
        from keras.models import Sequential
        from keras.layers import LSTM, Dense, Dropout, Bidirectional
        
        # 創建簡化的模型架構（用於緊急情況）
        model = Sequential([
            Bidirectional(LSTM(64, return_sequences=True, input_shape=(30, len(self.feature_names)))),
            Dropout(0.3),
            Bidirectional(LSTM(32, return_sequences=False)),
            Dropout(0.3),
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(3, activation='softmax')
        ])
        
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        # 嘗試載入權重
        try:
            # 如果有權重文件
            weights_path = self.model_path.replace('.h5', '_weights.h5')
            model.load_weights(weights_path)
            print("成功載入權重文件！")
            return model
        except:
            # 如果沒有單獨的權重文件，嘗試從完整模型提取權重
            print("警告: 無法載入原始模型，請考慮重新訓練")
            return None
    
    def safe_divide(self, numerator, denominator):
        """安全除法"""
        if isinstance(numerator, np.ndarray):
            numerator = pd.Series(numerator)
        if isinstance(denominator, np.ndarray):
            denominator = pd.Series(denominator)
        
        result = numerator / (denominator.abs() + 1e-6)
        result = result.replace([np.inf, -np.inf], np.nan)
        return result
    
    def compute_RSI(self, series, period=14):
        """計算RSI指標"""
        delta = series.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        rs = self.safe_divide(avg_gain, avg_loss)
        rsi = 100 - self.safe_divide(100, (1 + rs))
        return rsi.fillna(50)
    
    def compute_MACD(self, series, fast=12, slow=26, signal=9):
        """計算MACD指標"""
        ema_fast = series.ewm(span=fast).mean()
        ema_slow = series.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        histogram = macd - signal_line
        return macd, signal_line, histogram
    
    def compute_bollinger_bands(self, series, period=20, std_dev=2):
        """計算布林帶"""
        ma = series.rolling(period).mean()
        std = series.rolling(period).std()
        upper = ma + (std * std_dev)
        lower = ma - (std * std_dev)
        return upper, lower, ma
    
    def compute_KDJ(self, df, n=9, k_period=3, d_period=3):
        """計算KDJ指標"""
        low_min = df['Low'].rolling(n).min()
        high_max = df['High'].rolling(n).max()
        rsv = 100 * self.safe_divide((df['Close'] - low_min), (high_max - low_min))
        k = rsv.ewm(com=(k_period - 1), adjust=False).mean()
        d = k.ewm(com=(d_period - 1), adjust=False).mean()
        j = 3 * k - 2 * d
        return k, d, j
    
    def compute_williams_r(self, df, period=14):
        """計算Williams R指標"""
        high_max = df['High'].rolling(period).max()
        low_min = df['Low'].rolling(period).min()
        wr = -100 * self.safe_divide((high_max - df['Close']), (high_max - low_min))
        return wr
    
    def compute_ATR(self, df, period=14):
        """計算ATR指標"""
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        return true_range.rolling(period).mean()
    
    def compute_obv(self, df):
        """計算OBV指標"""
        obv = [0]
        for i in range(1, len(df)):
            if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
                obv.append(obv[-1] + df['Volume'].iloc[i])
            elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
                obv.append(obv[-1] - df['Volume'].iloc[i])
            else:
                obv.append(obv[-1])
        return pd.Series(obv, index=df.index)
    
    def fetch_latest_data(self, end_time, days_back=200):
        """獲取最新的市場數據"""
        print("正在下載最新市場數據...")
        end_date = end_time.strftime('%Y-%m-%d')
        start_date = (end_time - timedelta(days=days_back)).strftime('%Y-%m-%d')
        print(end_time)
        try:
            # 台積電數據
            stock_data = yf.download("2330.TW", start=start_date, end=end_date)
            if isinstance(stock_data.columns, pd.MultiIndex):
                stock_data.columns = stock_data.columns.get_level_values(0)
            
            # SOX指數
            sox_data = yf.download("SOXX", start=start_date, end=end_date)
            if isinstance(sox_data.columns, pd.MultiIndex):
                sox_data.columns = sox_data.columns.get_level_values(0)
            
            # 台股加權指數
            taiwan_index = yf.download("^TWII", start=start_date, end=end_date)
            if isinstance(taiwan_index.columns, pd.MultiIndex):
                taiwan_index.columns = taiwan_index.columns.get_level_values(0)
            
            # 美元指數
            usd_index = yf.download("DX-Y.NYB", start=start_date, end=end_date)
            if isinstance(usd_index.columns, pd.MultiIndex):
                usd_index.columns = usd_index.columns.get_level_values(0)
            
            # 台幣匯率
            try:
                usdtwd = yf.download("USDTWD=X", start=start_date, end=end_date)
            except:
                usdtwd = yf.download("TWD=X", start=start_date, end=end_date)
            
            if isinstance(usdtwd.columns, pd.MultiIndex):
                usdtwd.columns = usdtwd.columns.get_level_values(0)
            
            print("市場數據下載完成！")
            return stock_data, sox_data, taiwan_index, usd_index, usdtwd
            
        except Exception as e:
            print(f"數據下載失敗: {str(e)}")
            return None, None, None, None, None
    
    def compute_features(self, stock_data, sox_data, taiwan_index, usd_index, usdtwd): # 2330價錢 sox價錢 加權指數 美元指數 台幣指數
        """計算所有特徵"""
        print("正在計算技術指標...")
        
        # 確保所有數據都去除NaN
        for data in [stock_data, sox_data, taiwan_index, usd_index, usdtwd]:
            data.dropna(inplace=True)
        
        # 找到共同的日期範圍
        common_dates = stock_data.index.intersection(sox_data.index)\
                                      .intersection(taiwan_index.index)\
                                      .intersection(usd_index.index)\
                                      .intersection(usdtwd.index)
        
        if len(common_dates) < 100:
            print(f"警告: 共同日期數量不足 ({len(common_dates)})")
        
        # 對齊所有數據到共同日期
        stock_data = stock_data.loc[common_dates]
        sox_data = sox_data.loc[common_dates]
        taiwan_index = taiwan_index.loc[common_dates]
        usd_index = usd_index.loc[common_dates]
        usdtwd = usdtwd.loc[common_dates]

        
        # 初始化特徵DataFrame
        features = pd.DataFrame(index=common_dates)
        
        try:
            # 基礎價格特徵
            features['Returns_1'] = stock_data['Close'].pct_change(1)
            features['Returns_3'] = stock_data['Close'].pct_change(3)
            features['Returns_5'] = stock_data['Close'].pct_change(5)
            features['Returns_10'] = stock_data['Close'].pct_change(10)
            
            # 移動平均
            periods = [5, 10, 20, 50, 100]
            for period in periods:
                features[f'MA{period}'] = stock_data['Close'].rolling(period).mean()
                features[f'EMA{period}'] = stock_data['Close'].ewm(span=period).mean()
                features[f'MA{period}_slope'] = features[f'MA{period}'].diff(5)
                features[f'Price_MA{period}_ratio'] = self.safe_divide(stock_data['Close'], features[f'MA{period}'])
                if period <= 50:  # 只計算部分EMA比率
                    features[f'Price_EMA{period}_ratio'] = self.safe_divide(stock_data['Close'], features[f'EMA{period}'])
            
            # MA交叉信號
            features['MA5_MA20_cross'] = np.where(features['MA5'] > features['MA20'], 1, -1)
            features['MA10_MA50_cross'] = np.where(features['MA10'] > features['MA50'], 1, -1)
            features['MA20_MA100_cross'] = np.where(features['MA20'] > features['MA100'], 1, -1)
            
            # RSI指標
            for period in [7, 14, 21, 28]:
                features[f'RSI{period}'] = self.compute_RSI(stock_data['Close'], period)
                if period == 14:
                    features['RSI14_signal'] = np.where(features['RSI14'] > 70, -1, 
                                                       np.where(features['RSI14'] < 30, 1, 0))
                    features['RSI14_divergence'] = 0  # 簡化處理
            
            # MACD指標
            for fast, slow, signal in [(12, 26, 9), (5, 35, 5), (19, 39, 9)]:
                macd_line, signal_line, histogram = self.compute_MACD(stock_data['Close'], fast, slow, signal)
                suffix = f"_{fast}_{slow}_{signal}"
                features[f'MACD{suffix}'] = macd_line
                if fast == 12:  # 只保存標準MACD的信號線和柱狀圖
                    features[f'MACD_signal{suffix}'] = signal_line
                    features[f'MACD_histogram{suffix}'] = histogram
                    features[f'MACD_cross{suffix}'] = np.where(macd_line > signal_line, 1, -1)
            
            # 布林帶
            for period, std_dev in [(20, 2), (20, 2.5), (10, 1.5)]:
                bb_upper, bb_lower, bb_middle = self.compute_bollinger_bands(stock_data['Close'], period, std_dev)
                suffix = f"_{period}_{std_dev}"
                features[f'BB_width{suffix}'] = self.safe_divide((bb_upper - bb_lower), bb_middle)
                features[f'BB_position{suffix}'] = self.safe_divide((stock_data['Close'] - bb_lower), (bb_upper - bb_lower))
                if period == 20 and std_dev == 2:
                    features['BB_squeeze_20_2'] = (features['BB_width_20_2'] < features['BB_width_20_2'].rolling(20).mean() * 0.8).astype(int)
            
            # KDJ指標
            for n, k_period, d_period in [(9, 3, 3), (14, 5, 5)]:
                k, d, j = self.compute_KDJ(stock_data, n, k_period, d_period)
                suffix = f"_{n}_{k_period}_{d_period}"
                features[f'K{suffix}'] = k
                features[f'D{suffix}'] = d
                if n == 9:
                    features[f'J{suffix}'] = j
                    features[f'KD_cross{suffix}'] = np.where(k > d, 1, -1)
            
            # 其他技術指標
            features['Williams_R'] = self.compute_williams_r(stock_data)
            features['CCI'] = 0  # 簡化處理
            features['Stoch_RSI'] = 0  # 簡化處理
            features['ADX'] = 0  # 簡化處理
            
            # 成交量分析
            volume_ma20 = stock_data['Volume'].rolling(20).mean()
            features['Volume_ratio'] = self.safe_divide(stock_data['Volume'], volume_ma20)
            features['Volume_trend'] = stock_data['Volume'].rolling(5).mean() / volume_ma20 - 1
            features['Volume_price_trend'] = np.where(
                (features['Returns_5'] > 0) & (features['Volume_ratio'] > 1.2), 1,
                np.where((features['Returns_5'] < 0) & (features['Volume_ratio'] > 1.2), -1, 0)
            )
            features['Price_Volume_correlation'] = stock_data['Close'].rolling(10).corr(stock_data['Volume']).fillna(0)
            
            # OBV指標
            obv = self.compute_obv(stock_data)
            obv_ma = obv.rolling(20).mean()
            features['OBV_signal'] = np.where(obv > obv_ma, 1, -1)
            
            # ATR和波動率
            for period in [7, 14, 21, 30]:
                atr = self.compute_ATR(stock_data, period)
                features[f'ATR{period}_ratio'] = self.safe_divide(atr, stock_data['Close'])
                features[f'Volatility{period}'] = features['Returns_1'].rolling(period).std()
            
            # 價格位置分析
            features['High_Low_ratio'] = self.safe_divide((stock_data['High'] - stock_data['Low']), stock_data['Close'])
            features['Close_position'] = self.safe_divide((stock_data['Close'] - stock_data['Low']), (stock_data['High'] - stock_data['Low']))
            features['Upper_shadow'] = self.safe_divide((stock_data['High'] - np.maximum(stock_data['Open'], stock_data['Close'])), stock_data['Close'])
            features['Lower_shadow'] = self.safe_divide((np.minimum(stock_data['Open'], stock_data['Close']) - stock_data['Low']), stock_data['Close'])
            
            # 形態識別（簡化）
            features['Doji'] = ((stock_data['Close'] - stock_data['Open']).abs() / (stock_data['High'] - stock_data['Low']) < 0.1).astype(int)
            features['Hammer'] = 0  # 簡化處理
            features['Engulfing'] = 0  # 簡化處理
            
            # 趨勢分析
            features['Trend_strength'] = 0.5  # 簡化處理
            features['Price_velocity'] = stock_data['Close'].diff() / stock_data['Close'].shift()
            features['Price_acceleration'] = features['Price_velocity'].diff()
            
            # 支撐阻力位
            for window in [10, 20, 50]:
                resistance = stock_data['High'].rolling(window).max()
                support = stock_data['Low'].rolling(window).min()
                features[f'Resistance_distance_{window}'] = self.safe_divide((resistance - stock_data['Close']), stock_data['Close'])
                features[f'Support_distance_{window}'] = self.safe_divide((stock_data['Close'] - support), stock_data['Close'])
            
            # 市場相關特徵
            # SOX相關
            sox_returns = sox_data['Close'].pct_change()
            features['SOX_Returns_1'] = sox_returns
            features['SOX_Returns_5'] = sox_data['Close'].pct_change(5)
            features['SOX_RSI'] = self.compute_RSI(sox_data['Close'])
            sox_ma20 = sox_data['Close'].rolling(20).mean()
            features['SOX_Momentum'] = sox_data['Close'] / sox_ma20 - 1
            features['SOX_Volatility'] = sox_returns.rolling(20).std()
            
            # 台股相關
            twii_returns = taiwan_index['Close'].pct_change()
            features['TWII_Returns_1'] = twii_returns
            features['TWII_Returns_5'] = taiwan_index['Close'].pct_change(5)
            features['TWII_RSI'] = self.compute_RSI(taiwan_index['Close'])
            twii_ma20 = taiwan_index['Close'].rolling(20).mean()
            features['TWII_Momentum'] = taiwan_index['Close'] / twii_ma20 - 1
            features['TWII_Volatility'] = twii_returns.rolling(20).std()
            
            # 美元和匯率
            usd_returns = usd_index['Close'].pct_change()
            twd_returns = usdtwd['Close'].pct_change()
            features['USD_Returns_1'] = usd_returns
            features['USD_Volatility'] = usd_returns.rolling(20).std()
            features['USDTWD_Returns_1'] = twd_returns
            
            # 相關性特徵
            stock_returns = features['Returns_1']
            features['stock_SOX_Corr'] = stock_returns.rolling(20).corr(sox_returns)
            features['stock_TWII_Corr'] = stock_returns.rolling(20).corr(twii_returns)
            features['stock_USD_Corr'] = stock_returns.rolling(20).corr(usd_returns)
            features['stock_TW_Corr'] = stock_returns.rolling(20).corr(twd_returns)
            
            # 相對強度
            features['stock_SOX_Ratio'] = self.safe_divide(stock_data['Close'], sox_data['Close'])
            features['stock_TWII_Ratio'] = self.safe_divide(stock_data['Close'], taiwan_index['Close'])
            features['stock_vs_SOX'] = stock_returns - sox_returns
            features['stock_vs_TWII'] = stock_returns - twii_returns
            
            print("技術指標計算完成！")
            
            # 數據清理
            for col in features.columns:
                features[col] = features[col].replace([np.inf, -np.inf], np.nan)
                features[col] = features[col].fillna(method='ffill').fillna(0)
            
            return features # 各種技術指標
            
        except Exception as e:
            print(f"特徵計算錯誤: {str(e)}")
            return None
    
    def predict_tomorrow(self, end_time, confidence_threshold=0.75, ):
        """預測明天的股價走勢"""
        if not self.scaler or not self.model:
            print("請先載入模型和標準化器！")
            return None
        
        # 獲取最新數據
        stock_data, sox_data, taiwan_index, usd_index, usdtwd = self.fetch_latest_data(end_time=end_time, days_back=200)
        
        if stock_data is None:
            print("無法獲取市場數據！")
            return None
        
        # 計算特徵
        features_df = self.compute_features(stock_data, sox_data, taiwan_index, usd_index, usdtwd)
        
        if features_df is None:
            print("特徵計算失敗！")
            return None
        
        # 確保特徵與訓練時一致
        existing_features = [f for f in self.feature_names if f in features_df.columns]
        missing_features = [f for f in self.feature_names if f not in features_df.columns]
        
        if missing_features:
            print(f"缺少特徵: {len(missing_features)}個")
            # 用0填充缺少的特徵
            for feature in missing_features:
                features_df[feature] = 0
        
        # 選擇特徵並排序
        feature_data = features_df[self.feature_names].copy()
        print(feature_data)
        # 標準化特徵
        try:
            scaled_features = self.scaler.transform(feature_data)
        except Exception as e:
            print(f"特徵標準化失敗: {str(e)}")
            return None
        
        # 創建序列（使用最近30天的數據）
        time_steps = 30
        if len(scaled_features) < time_steps:
            print(f"數據不足，需要至少{time_steps}天的數據！")
            return None
        
        # 取最後30天作為預測序列
        sequence = scaled_features[-time_steps:].reshape(1, time_steps, len(self.feature_names))
        
        # 進行預測
        try:
            prediction_probs = self.model.predict(sequence, verbose=0)[0]
            predicted_class = np.argmax(prediction_probs)
            confidence = np.max(prediction_probs)
            
            # 獲取最新收盤價
            latest_close = stock_data['Close'].iloc[-1]
            
            # 類別映射
            class_names = ['跌', '觀望', '漲']
            prediction_name = class_names[predicted_class]
            
            # 準備結果
            result = {
                'prediction': prediction_name,
                'prediction_class': int(predicted_class),
                'confidence': float(confidence),
                'probabilities': {
                    '跌': float(prediction_probs[0]),
                    '觀望': float(prediction_probs[1]),
                    '漲': float(prediction_probs[2])
                },
                'latest_close': float(latest_close),
                'prediction_date': end_time.strftime('%Y-%m-%d'),
                'data_date': stock_data.index[-1].strftime('%Y-%m-%d'),
                'high_confidence': bool(confidence > confidence_threshold)
            }
            
            return result
            
        except Exception as e:
            print(f"預測失敗: {str(e)}")
            return None
    
    def print_prediction_report(self, result):
        """打印預測報告"""
        if not result:
            print("無預測結果可顯示")
            return
        
        print("\n" + "="*50)
        print("🏛️  台積電股價預測報告")
        print("="*50)
        
        print(f"📅 預測日期: {result['prediction_date']}")
        print(f"📊 數據截至: {result['data_date']}")
        print(f"💰 最新收盤價: NT$ {result['latest_close']:.2f}")
        
        print(f"\n🔮 明天預測: {result['prediction']}")
        print(f"📈 預測置信度: {result['confidence']:.1%}")
        print(f"⚡ 高置信度預測: {'是' if result['high_confidence'] else '否'}")
        
        print("\n📊 各類別機率:")
        for class_name, prob in result['probabilities'].items():
            emoji = "📈" if class_name == "漲" else "📉" if class_name == "跌" else "⏸️"
            print(f"  {emoji} {class_name}: {prob:.1%}")
        
        print(f"\n💡 投資建議:")
        if result['high_confidence']:
            if result['prediction'] == '漲':
                print("  🟢 模型顯示高置信度看漲，可考慮適當買入")
            elif result['prediction'] == '跌':
                print("  🔴 模型顯示高置信度看跌，建議謹慎操作")
            else:
                print("  🟡 模型建議觀望，等待更明確信號")
        else:
            print("  ⚠️  模型置信度較低，建議結合其他分析")
        
        print("\n⚠️  風險提醒:")
        print("  • 本預測僅供參考，不構成投資建議")
        print("  • 股市有風險，投資需謹慎")
        print("  • 請結合基本面分析和市場環境判斷")
        print("="*50)

    def join_txt2msg(self, result):
        msg=""
        msg+="🌕 Lymetix 例行報告\n\n"
        msg+="🏛️ 台積電股價預測報告"
        
        msg+=f"\n📅 預測日期: {result['prediction_date']}"
        msg+=f"\n💰 最新收盤價: NT$ {result['latest_close']:.2f}"
        
        msg+=f"\n\n🔮 今天預測: {result['prediction']}"
        msg+=f"\n📈 預測置信度: {result['confidence']:.1%}"
        msg+=f"\n⚡ 高置信度預測: {'是' if result['high_confidence'] else '否'}"
        
        msg += "\n\n📊 各類別機率:"
        for class_name, prob in result['probabilities'].items():
            emoji = "📈" if class_name == "漲" else "📉" if class_name == "跌" else "⏸️"
            msg+=f"\n  • {emoji} {class_name}: {prob:.1%}"
        
        msg+="\n\n💡 投資建議:"
        if result['high_confidence']:
            if result['prediction'] == '漲':
                msg+="\n  • 模型顯示高置信度看漲，可考慮適當買入"
            elif result['prediction'] == '跌':
                msg+="\n  • 模型顯示高置信度看跌，建議謹慎操作"
            else:
                msg+="\n  • 模型建議觀望，等待更明確信號"
        else:
            msg+="\n  ⚠️  模型置信度較低，建議結合其他分析"
        
        msg+="\n\n⚠️  風險提醒:"
        msg+="\n  • 本預測僅供參考，不構成投資建議"
        msg+="\n  • 股市有風險，投資需謹慎"
        msg+="\n  • 請結合基本面分析和市場環境判斷"
        return msg

    def backtest_predictions(self, end_time, days_back=4000):
        if not self.scaler or not self.model:
            print("請先載入模型和標準化器！")
            return None
        
        all_results = []
        
        
        # 獲取回測所需的全部數據
        # 這裡假設您的 fetch_latest_data 函式可以處理較長的 period
        # 如果您的 fetch_latest_data 函式無法處理，您需要修改它
        stock_data, sox_data, taiwan_index, usd_index, usdtwd = self.fetch_latest_data(
            end_time=end_time,
            days_back=days_back # 確保取得足夠的數據來計算特徵
        )
        
        if stock_data is None:
            print("無法獲取市場數據！")
            return None
        # 計算所有數據的特徵
        features_df = self.compute_features(stock_data, sox_data, taiwan_index, usd_index, usdtwd)
        print(features_df)
        
        if features_df is None:
            print("特徵計算失敗！")
            return None
            
        # 確保特徵與訓練時一致
        existing_features = [f for f in self.feature_names if f in features_df.columns]
        missing_features = [f for f in self.feature_names if f not in features_df.columns]
        
        if missing_features:
            print(f"缺少特徵: {len(missing_features)}個")
            for feature in missing_features:
                features_df[feature] = 0
                
        feature_data = features_df[self.feature_names].copy()
        
        # 標準化所有特徵
        try:
            scaled_features = self.scaler.transform(feature_data)
        except Exception as e:
            print(f"特徵標準化失敗: {str(e)}")
            return None
            
        time_steps = 30
        if len(scaled_features) < time_steps:
            print(f"數據不足，需要至少{time_steps}天的數據！")
            return None
        
        # 核心回測迴圈
        # 從第 time_steps 天開始，每天都進行一次預測
        for i in range(time_steps, len(scaled_features)):
            print("正在預測日期:", stock_data.index[i+146].strftime('%Y-%m-%d'), i)
            # 取得當前日期的序列數據
            sequence = scaled_features[i - time_steps : i].reshape(1, time_steps, len(self.feature_names))
            
            # 進行預測
            try:
                prediction_probs = self.model.predict(sequence, verbose=0)[0]

                # 獲取實際走勢
                actual_close_yesterday = stock_data['Close'].iloc[i +146 - 1]
                actual_close_today = stock_data['Close'].iloc[i+146]

                actual_class = -1
                
                # 設定漲跌閾值為 1.5%
                price_change_threshold_percent = 0.015
                price_change_percent = (actual_close_today - actual_close_yesterday) / actual_close_yesterday
                
                if price_change_percent > price_change_threshold_percent:
                    actual_class = 2
                elif price_change_percent < -price_change_threshold_percent:
                    actual_class = 0
                else:
                    actual_class = 1
                
                result = {
                    "date": stock_data.index[i+146].strftime('%Y-%m-%d'),
                    'probabilities': {
                        '跌': float(prediction_probs[0]),
                        '觀望': float(prediction_probs[1]),
                        '漲': float(prediction_probs[2])
                    },
                    "predict_class": int(np.argmax(prediction_probs)),
                    'actual_class': actual_class,
                    "actual_close_today": float(actual_close_today),
                }
                all_results.append(result)
                
            except Exception as e:
                print(f"日期 {stock_data.index[i+146].strftime('%Y-%m-%d')} 預測失敗: {str(e)}")
                continue # 遇到錯誤時跳過，繼續下一次迴圈
        
        return all_results


def main():
    """主程序示例"""
    print("🚀 台積電股價預測系統啟動")
    
    # 創建預測器實例
    predictor = stockStockPredictor()
    
    # 載入模型
    if not predictor.load_model_and_scaler():
        print("❌ 無法載入模型，請確認模型文件存在")
        print("需要的文件:")
        print(f"  - {SCALER_NAME}")
        print(f"  - {MODEL_NAME}")
        return
    
    # 進行預測
    print("\n正在進行預測分析...")

    result = predictor.predict_tomorrow(confidence_threshold=0.6, end_time=datetime.now())
    
    if result:
        # 打印預測報告
        predictor.print_prediction_report(result)
        analyzer = AdvancedAnalyzer(predictor)
            
            
        # 風險評估
        analyzer.risk_assessment(result)
        
        print(f"\n✅ 預測完成！建議結合基本面分析做最終決策。")
        return result
    else:
        print("❌ 預測失敗")
        return None

def make_test_report():
    predictor = stockStockPredictor()
    predictor.load_model_and_scaler()
    data = predictor.backtest_predictions(datetime.now(), days_back=6000)
    with open("result_data/back_train.json", "w", encoding="utf-8") as f:
        import json
        json.dump(data, f, ensure_ascii=False, indent=4)  
    

class AdvancedAnalyzer:
    """進階分析工具"""
    
    def __init__(self, predictor):
        self.predictor = predictor
    
    def analyze_prediction_history(self, days=10):
        """分析近期預測趨勢"""
        print(f"\n📊 近{days}天預測趨勢分析")
        print("注意: 此功能需要歷史預測數據")
        # 這裡可以擴展實現歷史預測追踪功能
        
    def risk_assessment(self, result):
        """風險評估"""
        if not result:
            return
            
        print("\n⚠️  風險評估:")
        
        # 基於置信度的風險評估
        if result['confidence'] > 0.8:
            risk_level = "低"
            color = "🟢"
        elif result['confidence'] > 0.6:
            risk_level = "中"
            color = "🟡"
        else:
            risk_level = "高"
            color = "🔴"
        
        print(f"  {color} 預測風險等級: {risk_level}")
        
        # 機率分散度分析
        probs = list(result['probabilities'].values())
        entropy = -sum(p * np.log(p + 1e-8) for p in probs)
        max_entropy = np.log(3)  # 3個類別的最大熵
        uncertainty = entropy / max_entropy
        
        print(f"  📊 預測不確定性: {uncertainty:.1%}")
        
        if uncertainty > 0.8:
            print("  ⚠️  模型對各種可能性都不太確定，建議謹慎操作")
        elif uncertainty < 0.3:
            print("  ✅ 模型預測相對明確")
        
        # 建議倉位大小
        if result['high_confidence'] and uncertainty < 0.5:
            if result['prediction'] in ['漲', '跌']:
                suggested_position = "可考慮適中倉位 (20-40%)"
            else:
                suggested_position = "建議觀望或輕倉 (5-10%)"
        else:
            suggested_position = "建議輕倉或觀望 (5-15%)"
        
        print(f"  💼 建議倉位: {suggested_position}")


def convert_model_to_weights_only(original_model_path):
    """將完整模型轉換為僅權重格式"""
    try:
        print("🔄 嘗試轉換模型為權重格式...")
        
        # 定義自定義函數
        def apply_attention_weight(inputs):
            features, weight = inputs
            weight_expanded = tf.expand_dims(weight, axis=1)
            return features * weight_expanded
        
        custom_objects = {'apply_attention_weight': apply_attention_weight}
        
        # 嘗試載入原模型
        model = load_model(original_model_path, custom_objects=custom_objects)
        
        # 保存權重
        weights_path = original_model_path.replace('.h5', '.weights.h5')
        model.save_weights(weights_path)
        print(f"✅ 權重已保存為: {weights_path}")
        
        # 保存模型架構(JSON格式)
        architecture_path = original_model_path.replace('.h5', '_architecture.json')
        with open(architecture_path, 'w') as f:
            f.write(model.to_json())
        print(f"📝 模型架構已保存為: {architecture_path}")
        
        return True, weights_path, architecture_path
        
    except Exception as e:
        print(f"❌ 模型轉換失敗: {str(e)}")
        return False, None, None

def make_predict_report(from_date=[2025, 8, 14], back_days=120):
    # 創建預測器實例
    predictor = stockStockPredictor()
    
    # 載入模型
    if not predictor.load_model_and_scaler():
        print("❌ 無法載入模型，請確認模型文件存在")
        print("需要的文件:")
        print(f"  - {SCALER_NAME}")
        print(f"  - {MODEL_NAME}")
        return
    
    # 進行預測
    print("\n正在進行預測分析...")
    all_result = []
    for time in range(1, 0, -1):
        print(f"  預測過去{time}天的走勢...")
        import datetime as dt
        end_time = dt.date(from_date[0],from_date[1],from_date[2]) - timedelta(days=time)
        result = predictor.predict_tomorrow(confidence_threshold=0.6, end_time=end_time)
        if result:
            all_result.append(result)
    print(all_result)
    with open("過去30天預測報告_跌.txt", "w", encoding="utf-8") as f:
        for item in all_result:
            f.write(str(item["probabilities"]["跌"])+ "\n")
    with open("過去30天預測報告_觀望.txt", "w", encoding="utf-8") as f:
        for item in all_result:
            f.write(str(item["probabilities"]["觀望"])+ "\n")
    with open("過去30天預測報告_漲.txt", "w", encoding="utf-8") as f:
        for item in all_result:
            f.write(str(item["probabilities"]["漲"])+ "\n")
    with open("過去30天預測報告_日期.txt", "w", encoding="utf-8") as f:
        for item in all_result:
            f.write(str(item["prediction_date"])+ "\n")
    with open("過去30天預測報告_實際.txt", "w", encoding="utf-8") as f:
        for item in all_result:
            f.write(str(item["latest_close"])+ "\n")



if __name__ == "__main__":
    # 首先嘗試修復模型載入問題
    print("🚀 台積電股價預測系統啟動")
    print("正在檢查模型載入問題...")
    
    # 嘗試修復模型載入問題
    try:
        # 方法1: 嘗試轉換現有模型
        success, weights_path, arch_path = convert_model_to_weights_only(MODEL_NAME)
        if success:
            print("✅ 模型轉換成功！")
        else:
            print("❌ 模型轉換成功！")
    except:
        print("⚠️ 模型修復失敗，將使用備用方案")


    result = main()
    # make_test_report()
    
    
    # precision 預測中多少正確  recall 正確中多少預測 f1-score   support
    