#!/usr/bin/env python
# coding: utf-8

# In[38]:


import pandas as pd

data = pd.read_csv('concat-filter.csv')

data.interpolate(method='linear', inplace=True)

data_dict = {}

for index, row in data.iterrows():
    station = row['station']
    year = row['year']
    if 1977 <= year <= 2020:
        if (station, year) not in data_dict:
            data_dict[(station, year)] = {'tavg': [], 'prcp': []}

        data_dict[(station, year)]['tavg'].append(row['tavg'])
        data_dict[(station, year)]['prcp'].append(row['prcp'])
    
data_dict


# In[39]:


# TODO：没处理N/S；只保留1977-2020; 有的站（1990，10）被（1990，1）匹配
result_data = []

for (station, year), data in data_dict.items():
    temperature = data['tavg']
    precipitation = data['prcp']
    climate_type = onehot[koppen(temperature, precipitation, 'N')]

    print(station, year)
    result_data.append({'station': station, 'year': year, 'type': climate_type})

result_df = pd.DataFrame(result_data, columns=['station', 'year', 'type'])

result_df.to_csv('classification.csv', index=False)


# ## LSTM
# 0.1607

# In[46]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# 1. 从CSV文件加载数据
data = pd.read_csv('classification.csv')

# 2. 数据准备
scaler = MinMaxScaler()
data['type'] = scaler.fit_transform(data['type'].values.reshape(-1, 1))
data = data.sort_values(by=['station', 'year'])

# 3. 分离数据为训练集和测试集
train_data, test_data = train_test_split(data, test_size=0.2, shuffle=False)

X_train = train_data['type'].values[:-1]
y_train = train_data['type'].values[1:]

X_train = X_train.reshape(-1, 1)
y_train = y_train.reshape(-1, 1)

# 4. 创建和训练LSTM模型
model = Sequential()
model.add(LSTM(units=50, activation='relu', input_shape=(1, 1)))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)

# 5. 预测2020年的类型
X_test = test_data['type'].values[:-1]
X_test = X_test.reshape(-1, 1)
y_test = test_data['type'].values[1:]
y_test = y_test.reshape(-1, 1)

y_pred = model.predict(X_test)

# 6. 计算整体的RMSE
rmse = mean_squared_error(y_test, y_pred, squared = False)

# 打印整体的RMSE
print("Overall RMSE:", rmse)


# In[13]:


import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error, f1_score
from sklearn.model_selection import train_test_split

# 1. 从CSV文件加载数据
data = pd.read_csv('classification.csv')

# 2. 数据准备
data = data.sort_values(by=['station', 'year'])

# 3. 分离数据为训练集和测试集
train_data, test_data = train_test_split(data, test_size=0.2, shuffle=False)

X_train = train_data['type'].values[:-1]
y_train = train_data['type'].values[1:]

X_train = X_train.reshape(-1, 1)
y_train = y_train.reshape(-1, 1)

# 4. 创建和训练LSTM模型
model = Sequential()
model.add(LSTM(units=50, activation='relu', input_shape=(1, 1)))
model.add(Dense(units=32, activation='softmax'))  # 32分类问题，使用softmax激活函数
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)

# 5. 预测2020年的类型
X_test = test_data['type'].values[:-1]
X_test = X_test.reshape(-1, 1)
y_test = test_data['type'].values[1:]

y_pred = model.predict(X_test)

# 6. 计算整体的RMSE（对于分类问题不是必需的）
# rmse = mean_squared_error(y_test, y_pred, squared=False)

# 7. 计算F1分数（多分类问题）
y_pred_classes = np.argmax(y_pred, axis=1)
f1 = f1_score(y_test, y_pred_classes, average='macro')

# 打印整体的F1分数
print("Overall F1 Score:", f1)


# ## ARIMA
# 0.3209

# In[50]:


import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# 1. 从CSV文件加载数据
data = pd.read_csv('classification.csv')

# 2. 数据准备
scaler = MinMaxScaler()
data['type'] = scaler.fit_transform(data['type'].values.reshape(-1, 1))
data = data.sort_values(by=['station', 'year'])

# 3. 分离数据为训练集和测试集
train_data, test_data = train_test_split(data, test_size=0.2, shuffle=False)

train_series = train_data['type']
test_series = test_data['type']

# 4. 拟合ARIMA模型
p, d, q = 40, 1, 1  # 根据您的时间序列分析结果选择适当的阶数
arima_model = ARIMA(train_series, order=(p, d, q))
arima_result = arima_model.fit()

# 5. 计算预测
forecast = arima_result.predict(start=len(train_series), end=len(train_series) + len(test_series) - 1, typ='levels')

# 6. 计算整体的RMSE
rmse = np.sqrt(mean_squared_error(test_series, forecast))

# 打印整体的RMSE
print("Overall RMSE:", rmse)


# ## CNN
# 0.2900
# 
# F1 = 0.2945

# In[60]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# 1. 从CSV文件加载数据
data = pd.read_csv('classification.csv')

# 2. 数据准备
scaler = MinMaxScaler()
data['type'] = scaler.fit_transform(data['type'].values.reshape(-1, 1))

# 3. 分离数据为训练集和测试集
train_data = data[data['year'] < 2020]  # 使用年份小于2020的数据作为训练集
test_data = data[data['year'] == 2020]  # 使用年份等于2020的数据作为测试集

X_train = train_data['type'].values[:-1]
y_train = train_data['type'].values[1:]

X_train = X_train.reshape(-1, 1)
y_train = y_train.reshape(-1, 1)

# 4. 创建和训练CNN模型
model = Sequential()
model.add(Conv1D(filters=8, kernel_size=1, activation='relu', input_shape=(1, 1)))
model.add(MaxPooling1D(pool_size=1))
model.add(Flatten())
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)

# 5. 预测2020年的类型
X_test = test_data['type'].values[:-1]
X_test = X_test.reshape(-1, 1)
y_test = test_data['type'].values[1:]
y_test = y_test.reshape(-1, 1)

y_pred = model.predict(X_test)

# 6. 计算整体的RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# 打印整体的RMSE
print("Overall RMSE:", rmse)


# In[12]:


import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

# 1. 从CSV文件加载数据
data = pd.read_csv('classification.csv')

# 2. 数据准备
data = data.sort_values(by=['station', 'year'])

# 3. 分离数据为训练集和测试集
train_data, test_data = train_test_split(data, test_size=0.2, shuffle=False)

X_train = train_data['type'].values[:-1]
y_train = train_data['type'].values[1:]

X_train = X_train.reshape(-1, 1)
y_train = y_train.reshape(-1, 1)

# 4. 创建和训练CNN模型
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=1, activation='relu', input_shape=(1, 1)))
model.add(MaxPooling1D(pool_size=1))
model.add(Flatten())
model.add(Dense(units=32, activation='softmax'))  # 32分类问题，使用softmax激活函数
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)

# 5. 预测2020年的类型
X_test = test_data['type'].values[:-1]
X_test = X_test.reshape(-1, 1)
y_test = test_data['type'].values[1:]

y_pred = model.predict(X_test)

# 6. 计算F1分数（多分类问题）
y_pred_classes = np.argmax(y_pred, axis=1)
f1 = f1_score(y_test, y_pred_classes, average='macro')

# 打印整体的F1分数
print("Overall F1 Score:", f1)


# ## 多层感知机MLP
# 0.2896
# 
# F1 = 

# In[61]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# 1. 从CSV文件加载数据
data = pd.read_csv('classification.csv')

# 2. 数据准备
scaler = MinMaxScaler()
data['type'] = scaler.fit_transform(data['type'].values.reshape(-1, 1))

# 3. 分离数据为训练集和测试集
train_data = data[data['year'] < 2020]  # 使用年份小于2020的数据作为训练集
test_data = data[data['year'] == 2020]  # 使用年份等于2020的数据作为测试集

X_train = train_data['type'].values[:-1]
y_train = train_data['type'].values[1:]

X_train = X_train.reshape(-1, 1)
y_train = y_train.reshape(-1, 1)

# 4. 创建和训练MLP模型
model = Sequential()
model.add(Dense(units=64, activation='relu', input_shape=(1,)))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)

# 5. 预测2020年的类型
X_test = test_data['type'].values[:-1]
X_test = X_test.reshape(-1, 1)
y_test = test_data['type'].values[1:]
y_test = y_test.reshape(-1, 1)

y_pred = model.predict(X_test)

# 6. 计算整体的RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# 打印整体的RMSE
print("Overall RMSE:", rmse)


# In[11]:


import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

# 1. 从CSV文件加载数据
data = pd.read_csv('classification.csv')

# 2. 数据准备
data = data.sort_values(by=['station', 'year'])

# 3. 分离数据为训练集和测试集
train_data, test_data = train_test_split(data, test_size=0.2, shuffle=False)

X_train = train_data['type'].values[:-1]
y_train = train_data['type'].values[1:]

X_train = X_train.reshape(-1, 1)
y_train = y_train.reshape(-1, 1)

# 4. 创建和训练MLP模型
model = Sequential()
model.add(Dense(units=64, activation='relu', input_dim=1))
model.add(Dense(units=32, activation='softmax'))  # 32分类问题，使用softmax激活函数
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)

# 5. 预测2020年的类型
X_test = test_data['type'].values[:-1]
X_test = X_test.reshape(-1, 1)
y_test = test_data['type'].values[1:]

y_pred = model.predict(X_test)

# 6. 计算F1分数（多分类问题）
y_pred_classes = np.argmax(y_pred, axis=1)
f1 = f1_score(y_test, y_pred_classes, average='macro')

# 打印整体的F1分数
print("Overall F1 Score:", f1)


# ## 线性回归LR

# In[63]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# 1. 从CSV文件加载数据
data = pd.read_csv('classification.csv')

# 2. 数据准备
scaler = MinMaxScaler()
data['type'] = scaler.fit_transform(data['type'].values.reshape(-1, 1))

# 3. 分离数据为训练集和测试集
train_data = data[data['year'] < 2020]  # 使用年份小于2020的数据作为训练集
test_data = data[data['year'] == 2020]  # 使用年份等于2020的数据作为测试集

X_train = train_data['type'].values[:-1]
y_train = train_data['type'].values[1:]

X_train = X_train.reshape(-1, 1)
y_train = y_train.reshape(-1, 1)

# 4. 创建和训练线性回归模型
model = LinearRegression()
model.fit(X_train, y_train)

# 5. 预测2020年的类型
X_test = test_data['type'].values[:-1]
X_test = X_test.reshape(-1, 1)
y_test = test_data['type'].values[1:]
y_test = y_test.reshape(-1, 1)

y_pred = model.predict(X_test)

# 6. 计算整体的RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# 打印整体的RMSE
print("Overall RMSE:", rmse)


# ## SVM

# In[89]:


import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

# 1. 从CSV文件加载数据
data = pd.read_csv('classification.csv')

# 2. 数据准备
data = data.sort_values(by=['station', 'year'])

# 3. 分离数据为训练集和测试集
train_data, test_data = train_test_split(data, test_size=0.2, shuffle=False)

X_train = train_data[['type']]  # 注意 X_train 是 DataFrame
y_train = train_data['type']

X_test = test_data[['type']]  # 注意 X_test 是 DataFrame
y_test = test_data['type']

# 4. 创建和训练 SVM 模型
model = SVC(kernel='linear', C=0.0001)  # 使用线性核，可以根据需要更改核函数和参数
model.fit(X_train, y_train)

# 5. 预测2020年的类型
y_pred = model.predict(X_test)

# 6. 计算 F1 分数（多分类问题）
f1 = f1_score(y_test, y_pred, average='micro')

# 打印整体的 F1 分数
print("Overall F1 Score:", f1)


# ## 朴素贝叶斯

# In[88]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score

# 1. 从CSV文件加载数据
data = pd.read_csv('classification.csv')

# 2. 数据准备
data = data.sort_values(by=['station', 'year'])

# 3. 分离数据为训练集和测试集
train_data, test_data = train_test_split(data, test_size=0.2, shuffle=False)

# 4. 合并训练和测试数据以进行一致的特征处理
combined_data = pd.concat([train_data, test_data])

# 5. 使用独热编码处理 "station" 列
combined_data = pd.get_dummies(combined_data, columns=['station'], drop_first=True)

# 6. 恢复分离的训练和测试数据
train_data = combined_data.iloc[:len(train_data), :]
test_data = combined_data.iloc[len(train_data):, :]

X_train = train_data.drop(['type'], axis=1)
y_train = train_data['type']

X_test = test_data.drop(['type'], axis=1)
y_test = test_data['type']

# 7. 创建和训练朴素贝叶斯模型
model = MultinomialNB()
model.fit(X_train, y_train)

# 8. 预测2020年的类型
y_pred = model.predict(X_test)

# 9. 计算F1分数（多分类问题）
f1 = f1_score(y_test, y_pred, average='micro')

# 打印整体的F1分数
print("Overall F1 Score:", f1)


# ## RNN

# In[5]:


import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from sklearn.metrics import mean_squared_error, f1_score
from sklearn.model_selection import train_test_split

# 1. 从CSV文件加载数据
data = pd.read_csv('classification.csv')

# 2. 数据准备
data = data.sort_values(by=['station', 'year'])

# 3. 分离数据为训练集和测试集
train_data, test_data = train_test_split(data, test_size=0.2, shuffle=False)

X_train = train_data['type'].values[:-1]
y_train = train_data['type'].values[1:]

X_train = X_train.reshape(-1, 1, 1)  # 添加时间步信息
y_train = y_train.reshape(-1, 1)

# 4. 创建和训练SimpleRNN模型
model = Sequential()
model.add(SimpleRNN(units=50, activation='relu', input_shape=(1, 1)))
model.add(Dense(units=32, activation='softmax'))  # 32分类问题，使用softmax激活函数
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)

# 5. 预测2020年的类型
X_test = test_data['type'].values[:-1]
X_test = X_test.reshape(-1, 1, 1)  # 添加时间步信息
y_test = test_data['type'].values[1:]

y_pred = model.predict(X_test)

# 6. 计算整体的RMSE（对于分类问题不是必需的）
# rmse = mean_squared_error(y_test, y_pred, squared=False)

# 7. 计算F1分数（多分类问题）
y_pred_classes = np.argmax(y_pred, axis=1)
f1 = f1_score(y_test, y_pred_classes, average='micro')

# 打印整体的F1分数
print("Overall F1 Score:", f1)


# ## DNN

# In[10]:


import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

# 1. 从CSV文件加载数据
data = pd.read_csv('classification.csv')

# 2. 数据准备
data = data.sort_values(by=['station', 'year'])

# 3. 分离数据为训练集和测试集
train_data, test_data = train_test_split(data, test_size=0.2, shuffle=False)

X_train = train_data['type'].values[:-1]
y_train = train_data['type'].values[1:]

X_train = X_train.reshape(-1, 1)
y_train = y_train.reshape(-1, 1)

# 4. 创建和训练深度神经网络模型
model = Sequential()
model.add(Dense(units=128, activation='relu', input_shape=(1, 1)))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=32, activation='softmax'))  # 32分类问题，使用softmax激活函数
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=1, batch_size=32, verbose=1)

# 5. 预测2020年的类型
X_test = test_data['type'].values[:-1]
X_test = X_test.reshape(-1, 1)
y_test = test_data['type'].values[1:]

y_pred = model.predict(X_test)

# 6. 计算F1分数（多分类问题）
y_pred_classes = np.argmax(y_pred, axis=1)
f1 = f1_score(y_test, y_pred_classes, average='micro')

# 打印整体的F1分数
print("Overall F1 Score:", f1)


# In[ ]:




