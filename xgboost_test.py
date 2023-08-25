from sklearn.datasets import load_iris
 
from xgboost.sklearn import XGBClassifier
from xgboost import plot_importance
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
 
 
def fitmodel(X, y , imgpath, class_num):
    #
    # Create training and test split
    #
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    print(len(x_train), len(x_test), len(y_train), len(y_test))
    
    # 数据集分割
    #x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=123457)
     
    clf = XGBClassifier(
        booster = 'gbtree',
        objective = 'multi:softmax',
        num_class = class_num,
        gamma = 0.1,
        max_depth = 6,
        reg_lambda = 2,
        subsample = 0.7,
        colsample_bytree = 0.7,
        min_child_weight = 3,
        eta = 0.1,
        seed = 1000,
        nthread = 4,
    )
     
    #训练模型
    clf.fit(x_train,y_train,eval_metric='auc')
     
    # 对测试集进行预测
    y_pred = clf.predict(x_test)
     
    #计算准确率
    accuracy = accuracy_score(y_test,y_pred)
    print('accuracy:%2.2f%%'%(accuracy*100))
     
    # 显示重要特征
    plot_importance(clf)
    # plt.show()
    plt.savefig(imgpath)
    plt.close()


# dataset=train1+train2 Pattern=Particle,Ring,Scratch,Normal
# 讀取xlsx文件
data = pd.read_excel("train_new.xlsx")  # 替換為你的Excel文件路徑
# 過濾掉欄位名為"PatternCode"且值為空的部分
data_filtered = data.dropna(subset=['PatternCode'])
print('data length', len(data))
print('data_filtered length', len(data_filtered))

# 假設你的X和y的列名稱分別為'feature_col_name'和'label_col_name'
X = data_filtered[['XSIZE', 'YSIZE', 'DEFECTAREA', 'DSIZE', 'FINEBINNUMBER', 'INTENSITY', 'XSIZE_1', 'YSIZE_1', 'DEFECTAREA_1', 'DSIZE_1', 'INTENSITY_1', 'DEFECTAREA_2', 'DSIZE_2', 'INTENSITY_2', 'XSIZE_2', 'YSIZE_2']]

y = data_filtered['PatternCode']  # 假設y是一個Series，包含標籤列
imgpath = 'xgboost_all_with_normal.png'
fitmodel(X, y , imgpath, 4)

# dataset=train1+train2 Pattern=Particle,Ring,Scratch
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
X = data_filtered[['XSIZE', 'YSIZE', 'DEFECTAREA', 'DSIZE', 'FINEBINNUMBER', 'INTENSITY', 'XSIZE_1', 'YSIZE_1', 'DEFECTAREA_1', 'DSIZE_1', 'INTENSITY_1', 'DEFECTAREA_2', 'DSIZE_2', 'INTENSITY_2', 'XSIZE_2', 'YSIZE_2']]

y = data_filtered['PatternCode']  # 假設y是一個Series，包含標籤列
imgpath = 'xgboost_all_without_normal.png'
fitmodel(X, y , imgpath, 3)

# dataset=train1 Pattern=Particle,Ring,Scratch,Normal
# 讀取xlsx文件
data = pd.read_excel("train_new.xlsx")  # 替換為你的Excel文件路徑
# 過濾掉欄位名為"PatternCode"且值為空的部分
data_filtered = data.dropna(subset=['PatternCode'])
print('data length', len(data))
print('data_filtered length', len(data_filtered))
# 留 Layer 等於 TRAIN1 的資料
data_filtered = data_filtered[data_filtered['Layer'] == 'TRAIN1']
print('data_filtered length', len(data_filtered))

# 假設你的X和y的列名稱分別為'feature_col_name'和'label_col_name'
X = data_filtered[['XSIZE', 'YSIZE', 'DEFECTAREA', 'DSIZE', 'FINEBINNUMBER', 'INTENSITY', 'XSIZE_1', 'YSIZE_1', 'DEFECTAREA_1', 'DSIZE_1', 'INTENSITY_1', 'DEFECTAREA_2', 'DSIZE_2', 'INTENSITY_2', 'XSIZE_2', 'YSIZE_2']]

y = data_filtered['PatternCode']  # 假設y是一個Series，包含標籤列
imgpath = 'xgboost_train1_with_normal.png'
fitmodel(X, y , imgpath, 4)

# dataset=train1 Pattern=Particle,Ring,Scratch
# 讀取xlsx文件
data = pd.read_excel("train_new.xlsx")  # 替換為你的Excel文件路徑
# 過濾掉欄位名為"PatternCode"且值為空的部分
data_filtered = data.dropna(subset=['PatternCode'])
print('data length', len(data))
print('data_filtered length', len(data_filtered))
# 過濾掉 PatternCode 等於 3 的資料
data_filtered = data_filtered[data_filtered['PatternCode'] != 3]
print('data_filtered length', len(data_filtered))
# 留 Layer 等於 TRAIN1 的資料
data_filtered = data_filtered[data_filtered['Layer'] == 'TRAIN1']
print('data_filtered length', len(data_filtered))

# 假設你的X和y的列名稱分別為'feature_col_name'和'label_col_name'
X = data_filtered[['XSIZE', 'YSIZE', 'DEFECTAREA', 'DSIZE', 'FINEBINNUMBER', 'INTENSITY', 'XSIZE_1', 'YSIZE_1', 'DEFECTAREA_1', 'DSIZE_1', 'INTENSITY_1', 'DEFECTAREA_2', 'DSIZE_2', 'INTENSITY_2', 'XSIZE_2', 'YSIZE_2']]

y = data_filtered['PatternCode']  # 假設y是一個Series，包含標籤列
imgpath = 'xgboost_train1_without_normal.png'
fitmodel(X, y , imgpath, 3)

# dataset=train2 Pattern=Particle,Ring,Scratch,Normal
# 讀取xlsx文件
data = pd.read_excel("train_new.xlsx")  # 替換為你的Excel文件路徑
# 過濾掉欄位名為"PatternCode"且值為空的部分
data_filtered = data.dropna(subset=['PatternCode'])
print('data length', len(data))
print('data_filtered length', len(data_filtered))
# 留 Layer 等於 TRAIN2 的資料
data_filtered = data_filtered[data_filtered['Layer'] == 'TRAIN2']
print('data_filtered length', len(data_filtered))

# 假設你的X和y的列名稱分別為'feature_col_name'和'label_col_name'
X = data_filtered[['XSIZE', 'YSIZE', 'DEFECTAREA', 'DSIZE', 'FINEBINNUMBER', 'INTENSITY', 'XSIZE_1', 'YSIZE_1', 'DEFECTAREA_1', 'DSIZE_1', 'INTENSITY_1', 'DEFECTAREA_2', 'DSIZE_2', 'INTENSITY_2', 'XSIZE_2', 'YSIZE_2']]

y = data_filtered['PatternCode']  # 假設y是一個Series，包含標籤列
imgpath = 'xgboost_train2_with_normal.png'
fitmodel(X, y , imgpath, 4)

# dataset=train2 Pattern=Particle,Ring,Scratch
# 讀取xlsx文件
data = pd.read_excel("train_new.xlsx")  # 替換為你的Excel文件路徑
# 過濾掉欄位名為"PatternCode"且值為空的部分
data_filtered = data.dropna(subset=['PatternCode'])
print('data length', len(data))
print('data_filtered length', len(data_filtered))
# 過濾掉 PatternCode 等於 3 的資料
data_filtered = data_filtered[data_filtered['PatternCode'] != 3]
print('data_filtered length', len(data_filtered))
# 留 Layer 等於 TRAIN2 的資料
data_filtered = data_filtered[data_filtered['Layer'] == 'TRAIN2']
print('data_filtered length', len(data_filtered))

# 假設你的X和y的列名稱分別為'feature_col_name'和'label_col_name'
X = data_filtered[['XSIZE', 'YSIZE', 'DEFECTAREA', 'DSIZE', 'FINEBINNUMBER', 'INTENSITY', 'XSIZE_1', 'YSIZE_1', 'DEFECTAREA_1', 'DSIZE_1', 'INTENSITY_1', 'DEFECTAREA_2', 'DSIZE_2', 'INTENSITY_2', 'XSIZE_2', 'YSIZE_2']]

y = data_filtered['PatternCode']  # 假設y是一個Series，包含標籤列
imgpath = 'xgboost_train2_without_normal.png'
fitmodel(X, y , imgpath, 3)
