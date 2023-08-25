#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt
import matplotlib

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


def fitmodel(X, y , imgpath, class_num):    
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    print(len(x_train), len(x_test), len(y_train), len(y_test))
    
    knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
    knn.fit(x_train, y_train)
    
    print('The accuracy of the knn classifier is {:.2f} out of 1 on training data'.format(knn.score(x_train, y_train)))
    print('The accuracy of the knn classifier is {:.2f} out of 1 on test data'.format(knn.score(x_test, y_test)))
    
    '''
    k_range = list(range(1,100))
    scores = []
    mean_scores = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(x, y)
        #print('The accuracy of the knn classifier is {:.2f} out of 1 on training data'.format(knn.score(x_train, y_train)))
        #print('The accuracy of the knn classifier is {:.2f} out of 1 on test data'.format(knn.score(x_test, y_test)))
    
        y_pred = knn.predict(x)
        scores.append(metrics.accuracy_score(y, y_pred))
        
        cross_scores = cross_val_score(knn, x, y, cv=2)  # 5 折交叉驗證
        mean_scores.append(cross_scores.mean())
        
    # 找到最佳的 n_neighbors 值
    best_n_neighbors = k_range[mean_scores.index(max(mean_scores))]
    best_score = scores[mean_scores.index(max(mean_scores))]
    print("Best n_neighbors:", best_n_neighbors)
    print("Best scores:", best_score)
    
    plt.plot(k_range, scores)
    plt.xlabel('Value of k for KNN')
    plt.ylabel('Accuracy Score')
    plt.title('Accuracy Scores for Values of k of k-Nearest-Neighbors')
    plt.show()
    '''
    
    # # logistic regression
    logreg = LogisticRegression()
    logreg.fit(X, y)
    y_pred = logreg.predict(X)
    print('The accuracy of the logreg classifier', metrics.accuracy_score(y, y_pred))
    
    
    # # svm support vector machine :
    ga = 1 / class_num
    print(ga)
    svm = SVC(kernel='rbf', random_state=0, gamma=ga, C=1.0)
    svm.fit(x_train, y_train)
    
    print('The accuracy of the svm classifier on training data is {:.2f} out of 1'.format(svm.score(x_train, y_train)))
    print('The accuracy of the svm classifier on test data is {:.2f} out of 1'.format(svm.score(x_test, y_test)))




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




