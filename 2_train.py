# 2_train.py
import pandas as pd
import numpy as np
import tensorflow as tf
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

def run():
    if not os.path.exists('gesture_data.csv'):
        print("錯誤：找不到 gesture_data.csv，請先執行 1_collect.py")
        return

    print("=== [步驟 2] 模型訓練模式 ===")
    # 讀取資料
    df = pd.read_csv('gesture_data.csv', header=None)
    X = df.iloc[:, 1:].values
    y = df.iloc[:, 0].values
    
    num_classes = len(np.unique(y)) 
    print(f"偵測到 {num_classes} 種手勢類別，開始訓練...")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # 建立模型
    model = Sequential([
        Dense(64, input_shape=(42,), activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test))
    
    # 儲存
    model.save('gesture_password_model.h5')
    print("\n訓練完成！模型已儲存為 'gesture_password_model.h5'")

if __name__ == "__main__":
    run()