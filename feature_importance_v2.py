import shap
import tensorflow as tf 
tf.compat.v1.disable_v2_behavior()
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import load_model


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

model = load_model('result/my_model_10_32_5000.keras')
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
explainer = shap.DeepExplainer(model, data=X_train)

# 使用 tf.keras 載入模型
#model = tf.keras.models.load_model('result/my_model_10_32_5000.keras')
#
## 將 Keras Sequential 模型轉換為 tf.keras 模型
#tf_model = tf.keras.models.clone_model(model)
#tf_model.set_weights(model.get_weights())

# 創建 TensorFlow 會話
#session = tf.compat.v1.keras.backend.get_session()
#
## 使用 tf.keras 模型和會話來創建解釋器
#explainer = shap.DeepExplainer((tf_model, tf_model.input), data=X_train)
#
# 計算 SHAP 值
shap_values = explainer.shap_values(X_new)

# 獲得各個特徵的SHAP值
print(shap_values)

# 將 shap_values 轉換為 DataFrame
shap_values_df = pd.DataFrame(shap_values, columns=feature_col_name)

# 將 shap_values_df 保存為 CSV 文件
shap_values_df.to_csv('shap_values.csv', index=False)