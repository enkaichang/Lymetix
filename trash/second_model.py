import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

class StockMetaPredictor:
    """
    股票預測後測模型
    用於接收主模型的機率輸出並提升整體準確度
    """
    
    def __init__(self):
        self.models = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'gradient_boost': GradientBoostingClassifier(random_state=42),
            'logistic': LogisticRegression(random_state=42, max_iter=1000),
            'neural_network': MLPClassifier(hidden_layer_sizes=(64, 32), random_state=42, max_iter=500)
        }
        self.scaler = StandardScaler()
        self.best_model = None
        self.best_model_name = None
        self.class_mapping = {'跌': 0, '觀望': 1, '漲': 2}
        self.reverse_mapping = {0: '跌', 1: '觀望', 2: '漲'}
        
    def load_data(self, json_file_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """載入JSON格式的訓練資料"""
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        features = []
        labels = []
        
        for item in data:
            # 提取機率特徵
            probs = item['probabilities']
            feature = [probs['跌'], probs['觀望'], probs['漲']]
            
            # 添加額外特徵
            # 1. 最大機率值
            max_prob = max(feature)
            # 2. 機率分佈的熵值（不確定性）
            entropy = -sum(p * np.log(p + 1e-10) for p in feature if p > 0)
            # 3. 最大機率與次大機率的差距
            sorted_probs = sorted(feature, reverse=True)
            prob_gap = sorted_probs[0] - sorted_probs[1]
            # 4. 主模型預測的類別（最大機率對應的類別）
            predicted_class = np.argmax(feature)
            
            # 組合所有特徵
            enhanced_feature = feature + [max_prob, entropy, prob_gap, predicted_class]
            features.append(enhanced_feature)
            labels.append(item['actual_class'])
        
        return np.array(features), np.array(labels)
    
    def create_features_from_dict(self, prob_dict: Dict[str, float]) -> np.ndarray:
        """從機率字典創建特徵向量"""
        feature = [prob_dict['跌'], prob_dict['觀望'], prob_dict['漲']]
        
        # 添加增強特徵
        max_prob = max(feature)
        entropy = -sum(p * np.log(p + 1e-10) for p in feature if p > 0)
        sorted_probs = sorted(feature, reverse=True)
        prob_gap = sorted_probs[0] - sorted_probs[1]
        predicted_class = np.argmax(feature)
        
        enhanced_feature = feature + [max_prob, entropy, prob_gap, predicted_class]
        return np.array(enhanced_feature).reshape(1, -1)
    
    def train_models(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """訓練所有模型並返回交叉驗證分數"""
        X_scaled = self.scaler.fit_transform(X)
        
        results = {}
        
        print("開始訓練各個模型...")
        print("=" * 50)
        
        for name, model in self.models.items():
            print(f"訓練 {name}...")
            
            # 交叉驗證
            cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')
            mean_score = cv_scores.mean()
            std_score = cv_scores.std()
            
            results[name] = mean_score
            print(f"{name} - 平均準確度: {mean_score:.4f} (+/- {std_score * 2:.4f})")
        
        # 選擇最佳模型
        best_name = max(results, key=results.get)
        self.best_model_name = best_name
        self.best_model = self.models[best_name]
        
        # 訓練最佳模型
        self.best_model.fit(X_scaled, y)
        
        print("=" * 50)
        print(f"最佳模型: {best_name} (準確度: {results[best_name]:.4f})")
        
        return results
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray):
        """評估模型性能"""
        X_test_scaled = self.scaler.transform(X_test)
        y_pred = self.best_model.predict(X_test_scaled)
        y_pred_proba = self.best_model.predict_proba(X_test_scaled)
        
        print("\n模型評估結果:")
        print("=" * 50)
        print(f"測試準確度: {accuracy_score(y_test, y_pred):.4f}")
        print("\n分類報告:")
        print(classification_report(y_test, y_pred, 
                                  target_names=['跌', '觀望', '漲']))
        
        # 混淆矩陣
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['跌', '觀望', '漲'],
                   yticklabels=['跌', '觀望', '漲'])
        plt.title('混淆矩陣')
        plt.ylabel('實際類別')
        plt.xlabel('預測類別')
        plt.show()
        
        return y_pred, y_pred_proba
    
    def predict(self, probabilities: Dict[str, float]) -> Dict[str, Any]:
        """對新數據進行預測"""
        if self.best_model is None:
            raise ValueError("模型尚未訓練，請先調用 train_models 方法")
        
        # 創建特徵
        X = self.create_features_from_dict(probabilities)
        X_scaled = self.scaler.transform(X)
        
        # 預測
        prediction = self.best_model.predict(X_scaled)[0]
        prediction_proba = self.best_model.predict_proba(X_scaled)[0]
        
        # 組織結果
        result = {
            'predicted_class': int(prediction),
            'predicted_label': self.reverse_mapping[prediction],
            'confidence': float(max(prediction_proba)),
            'probabilities': {
                '跌': float(prediction_proba[0]),
                '觀望': float(prediction_proba[1]),
                '漲': float(prediction_proba[2])
            },
            'original_probabilities': probabilities,
            'model_used': self.best_model_name
        }
        
        return result
    
    def analyze_feature_importance(self):
        """分析特徵重要性（僅適用於樹模型）"""
        if hasattr(self.best_model, 'feature_importances_'):
            feature_names = ['跌_機率', '觀望_機率', '漲_機率', '最大機率', 
                           '熵值', '機率差距', '主模型預測']
            
            importance = self.best_model.feature_importances_
            
            plt.figure(figsize=(10, 6))
            indices = np.argsort(importance)[::-1]
            plt.bar(range(len(importance)), importance[indices])
            plt.xticks(range(len(importance)), 
                      [feature_names[i] for i in indices], rotation=45)
            plt.title('特徵重要性分析')
            plt.tight_layout()
            plt.show()
            
            print("特徵重要性排序:")
            for i in indices:
                print(f"{feature_names[i]}: {importance[i]:.4f}")

# 使用範例
def main():
    # 創建預測器
    predictor = StockMetaPredictor()
    X, y = predictor.load_data('back_train.json')

    # 分割數據
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 訓練模型
    predictor.train_models(X_train, y_train)

