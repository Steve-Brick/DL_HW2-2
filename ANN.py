import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

if __name__ == '__main__':
    # 导入数据
    info = pd.read_csv("boston.csv")
    X = info.iloc[:, :-1] #函数的自变量(13维)
    y=info.iloc[:,-1] #函数的因变量(1维)

    # 构建神经网络
    model=tf.keras.Sequential([
        tf.keras.layers.Dense(13,activation='linear'),
        tf.keras.layers.Dense(4, activation='relu'),
        tf.keras.layers.Dense(1, activation='relu')
    ])
    model.compile(
        optimizer='adam',
        loss='MSE'
    )

    # 拟合函数
    history=model.fit(
        X, y,
        epochs=150,
        batch_size=64,
    )

    # 绘制损失函数随epoch的变化
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # 绘制拟合效果图
    y_pred = model.predict(X)
    plt.figure(figsize=(10, 6))
    plt.plot(y,label="Real",color='b')
    plt.plot(y_pred, label="Predict",color='r')
    plt.legend()
    plt.show()