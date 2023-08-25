#使用梯度下降方法進行特徵重要性評估
import numpy as np
import tensorflow as tf
#from keras import backend as K
from keras.models import load_model
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 載入並編譯模型
model = load_model('result/my_model_10_32_5000.keras')
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#feature_col_name
feature_col_name = ['XSIZE', 'YSIZE', 'DEFECTAREA', 'DSIZE', 'FINEBINNUMBER', 'INTENSITY', 'XSIZE_1', 'YSIZE_1', 'DEFECTAREA_1', 'DSIZE_1', 'INTENSITY_1', 'DEFECTAREA_2', 'DSIZE_2', 'INTENSITY_2', 'XSIZE_2', 'YSIZE_2']

# 假設你有一個新的數據 X_new

# 讀取xlsx文件
data = pd.read_excel("train_new.xlsx")  # 替換為你的Excel文件路徑

# 過濾掉欄位名為"PatternCode"且值為空的部分
data_filtered = data.dropna(subset=['PatternCode'])
print('data length', len(data))
print('data_filtered length', len(data_filtered))
# 過濾掉 PatternCode 等於 3 的資料
data_filtered = data_filtered[data_filtered['PatternCode'] != 3]
print('data_filtered length', len(data_filtered))

# 假設你的X和y的列名稱分別為'feature_col_name'和'label_col_name'
X_new = data_filtered[feature_col_name]
X_new.to_csv('X_new.csv', index=True)

y = data_filtered['PatternCode']  # 假設y是一個Series，包含標籤列
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.3, stratify=y, random_state=42)
print(len(X_train), len(X_test), len(y_train), len(y_test))

# 將 X_new 轉換為 TensorFlow 張量
X_new = tf.convert_to_tensor(X_new.values, dtype=tf.float32)

# 定義梯度計算函數
def get_gradients(model, inputs, outputs):
    with tf.GradientTape() as tape:
        tape.watch(inputs)
        predictions = model(inputs)
        gradients = tape.gradient(predictions, inputs)
    return gradients

# 獲取模型的輸出和梯度
outputs = model.output
gradients = get_gradients(model, X_new, outputs)

# 計算各個特徵的重要性得分（可以取絕對值或平方來獲得更直觀的重要性）
feature_importance = np.abs(gradients.numpy())
# 將 feature_importance 按照列（特徵）加總
sum_by_feature = np.sum(feature_importance, axis=0)
# 將 feature_importance 按照列（特徵）求平均
mean_by_feature = np.mean(feature_importance, axis=0)

# 獲得各個特徵的重要性得分
print(feature_importance)
feature_importance_df = pd.DataFrame(data=feature_importance, columns=feature_col_name)

# 將 feature_importance_df 存儲為 CSV 檔案
feature_importance_df.to_csv('feature_importance_raw.csv', index=False)

# 將特徵名稱與加總、平均、中位數重要性得分合併成 DataFrame
feature_importance_df = pd.DataFrame({
    'Feature': feature_col_name,
    'Sum': sum_by_feature,
    'Mean': mean_by_feature
})

feature_importance_df.to_csv('feature_importance.csv', index=False)


##使用SHAP值進行特徵重要性評估：
#import shap
#
## 創建SHAP解釋器
#explainer = shap.DeepExplainer(model, X_train)
#
## 計算SHAP值
#shap_values = explainer.shap_values(X_new)
#
## 獲得各個特徵的SHAP值
#print(shap_values)
#
## 將 shap_values 轉換為 DataFrame
#shap_values_df = pd.DataFrame(shap_values, columns=feature_col_name)
#
## 將 shap_values_df 保存為 CSV 文件
#shap_values_df.to_csv('shap_values.csv', index=False)

#Permutation Importance
import numpy as np
from sklearn.metrics import accuracy_score

# 假設你已經載入並編譯了你的神經網絡模型 model

# 計算基準準確度
y_pred_baseline = model.predict(X_test)
baseline_accuracy = accuracy_score(y_test, np.argmax(y_pred_baseline, axis=1))

# 初始化特徵重要性得分列表
feature_importance_scores = []

# 對每個特徵進行打亂測試
for feature_index in range(X_test.shape[1]):
    # 創建新的測試集，對要評估的特徵打亂值
    X_test_shuffled = X_test.copy()
    np.random.shuffle(X_test_shuffled.values[:, feature_index])
    
    # 使用打亂後的測試集進行預測
    y_pred_shuffled = model.predict(X_test_shuffled)
    shuffled_accuracy = accuracy_score(y_test, np.argmax(y_pred_shuffled, axis=1))
    
    # 計算特徵重要性得分
    feature_importance_score = baseline_accuracy - shuffled_accuracy
    feature_importance_scores.append(feature_importance_score)

# 獲得各個特徵的重要性得分
print(feature_importance_scores)
