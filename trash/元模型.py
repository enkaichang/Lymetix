import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
# 導入 SMOTE 過取樣技術
from imblearn.over_sampling import SMOTE

# 定義檔案名稱，請確保與 backtest_predictions() 函式中產生的檔名一致
DATA_FILE = "back_train.json"
META_MODEL_NAME = "meta_model_lgbm.pkl"

def train_meta_model_lgbm():
    """
    載入原始模型的歷史預測資料，訓練一個基於 LightGBM 的元模型。
    
    這個元模型以原始模型的預測機率為輸入特徵，以實際的漲跌結果為目標標籤。
    """
    print("--- 開始訓練 LightGBM 元模型 ---")

    # 檢查資料檔案是否存在
    if not os.path.exists(DATA_FILE):
        print(f"⚠️ 錯誤: 未找到資料檔案 '{DATA_FILE}'。")
        print("請先執行您的原始程式，確保 backtest_predictions() 函式已成功生成此檔案。")
        return

    # 載入歷史預測資料
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ 載入資料檔案時發生錯誤: {e}")
        return

    if not data:
        print("⚠️ 資料檔案為空，無法訓練元模型。")
        return

    print(f"✅ 成功載入 {len(data)} 筆歷史預測資料。")

    # 準備訓練資料集
    df = pd.DataFrame(data)

    required_cols = ['probabilities', 'actual_class']
    if not all(col in df.columns for col in required_cols):
        print(f"❌ 資料檔案中缺少必要的欄位。請確認 JSON 結構是否正確。")
        return

    # 提取並新增特徵
    X = df['probabilities'].apply(lambda x: pd.Series([x['跌'], x['觀望'], x['漲']]))
    X.columns = ['prob_fall', 'prob_flat', 'prob_rise']
    
    X['prev_prob_fall'] = X['prob_fall'].shift(1)
    X['prev_prob_flat'] = X['prob_flat'].shift(1)
    X['prev_prob_rise'] = X['prob_rise'].shift(1)
    
    X['prob_fall_ma3'] = X['prob_fall'].rolling(window=3).mean()
    X['prob_rise_ma3'] = X['prob_rise'].rolling(window=3).mean()
    
    X['prob_fall_ma5'] = X['prob_fall'].rolling(window=5).mean()
    X['prob_rise_ma5'] = X['prob_rise'].rolling(window=5).mean()
    
    X['prob_fall_ma7'] = X['prob_fall'].rolling(window=7).mean()
    X['prob_rise_ma7'] = X['prob_rise'].rolling(window=7).mean()

    X['prob_fall_change'] = X['prob_fall'].diff()
    X['prob_rise_change'] = X['prob_rise'].diff()

    # 在進行 dropna 之前，先創建 y
    y = df['actual_class']

    # 將 X 和 y 組合，然後再一起移除包含 NaN 的行
    print("\n--- 正在清理資料集 ---")
    initial_rows = len(X)
    combined_df = pd.concat([X, y.rename('actual_class')], axis=1)
    combined_df = combined_df.dropna()
    rows_after_dropna = len(combined_df)
    print(f"✅ 移除空值行後，剩餘 {rows_after_dropna} 筆資料。")

    # 重新分割為 X 和 y
    X = combined_df.drop('actual_class', axis=1)
    y = combined_df['actual_class']
    
    # 定義數值到類別名稱的映射
    class_map = {'跌': 0, '觀望': 1, '漲': 2}
    reverse_class_map = {v: k for k, v in class_map.items()}

    # 檢查 y 中是否有無效值，並移除對應的行
    valid_indices = y.isin(reverse_class_map.keys())
    X = X[valid_indices]
    y = y[valid_indices]
    rows_after_filtering = len(X)
    print(f"✅ 移除無效類別行後，剩餘 {rows_after_filtering} 筆資料。")

    if rows_after_filtering == 0:
        print("⚠️ 錯誤: 經過資料清理後，訓練資料集為空，無法進行訓練。")
        print("請檢查您的 back_tain.json 檔案，確保包含有效的預測資料。")
        return
    
    if len(y.unique()) < 3:
        print("⚠️ 資料中缺少部分類別（跌、觀望、漲），訓練可能無效。")

    # LightGBM 要求目標變數為數值型，因此進行編碼
    # 這裡 y 已經是數值，所以直接使用
    y_encoded = y
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    
    # 執行 SMOTE 過取樣
    print("\n--- 執行 SMOTE 過取樣以處理資料不平衡 ---")
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    print(f"✅ SMOTE 過取樣完成。訓練集大小從 {len(X_train)} 變為 {len(X_train_resampled)}。")
    print("各類別樣本數 (重新取樣後):")
    print(pd.Series(y_train_resampled).value_counts())

    # 定義 LightGBM 的超參數網格
    param_grid = {
        'n_estimators': [150, 250, 400],
        'max_depth': [3, 5, 7, 9],
        'learning_rate': [0.1, 0.05, 0.025],
        'num_leaves': [31, 50, 100]
    }

    meta_model = GridSearchCV(LGBMClassifier(random_state=42, n_jobs=-1, objective='multiclass', class_weight='balanced'), param_grid, cv=5, scoring='f1_weighted', verbose=1)
    
    meta_model.fit(X_train_resampled, y_train_resampled)
    print("✅ LightGBM 元模型訓練完成。")
    
    print("\n--- 超參數調校結果 ---")
    print(f"最佳參數: {meta_model.best_params_}")
    
    y_pred = meta_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # 將數值預測結果轉換回原始類別名稱
    y_pred_labels = pd.Series(y_pred).map(reverse_class_map)
    y_test_labels = y.map(reverse_class_map).loc[X_test.index]

    print("\n--- LightGBM 元模型評估報告 ---")
    print(f"測試集準確率: {accuracy:.4f}")
    print("\n詳細分類報告:")
    print(classification_report(y_test_labels, y_pred_labels, target_names=class_map.keys()))
    
    try:
        joblib.dump(meta_model.best_estimator_, META_MODEL_NAME)
        print(f"\n✅ LightGBM 元模型已成功儲存為 '{META_MODEL_NAME}'。")
    except Exception as e:
        print(f"\n❌ 儲存元模型時發生錯誤: {e}")