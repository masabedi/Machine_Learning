# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

# تغییر سایز تست ست به 10 عدد
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3 , random_state = 0)

# 
from sklearn.linear_model import LinearRegression
# ایجاد یک ماشین برای تحلیل و پیش بینی
regressor = LinearRegression()

# آموزش ماشین برای تحلیل بر روی ست آموزش و یافتن ارتباط بین داده ها
regressor.fit(X_train, y_train)

# ایجاد بردار پیش بینی
# این بردار نتایج تحلیل و ارتباط بدست آمده از روی ترین ست را 
# بر روی تست ست نمایش می دهد و می توان با استفاده از این بردار تفاوت پیشبینی و مقدار واقعی را مشاهده نمود 
y_pred = regressor.predict(X_test)

# نمایش نتایج ترین ست
plt.scatter(X_train, y_train, color = 'red')
# نمودار خط رگرسیون
# متغیر وابسته در اینجامقدار زیر خواهد بود 
# y = regressor.predict(X_train)
plt.plot(X_train, regressor.predict(X_train), color = 'blue')

# ایجاد عنوان
plt.title("salary vs experience(training set)")

# 
plt.xlabel("years of experience")
plt.ylabel("salary")

plt.show()


# نمایش نتایج تست ست
plt.scatter(X_test, y_test, color = 'red')
# نمودار خط رگرسیون
# متغیر وابسته در اینجامقدار زیر خواهد بود 
# y = regressor.predict(X_train)

plt.plot(X_train, regressor.predict(X_train), color = 'blue')

# ایجاد عنوان
plt.title("salary vs experience(test set)")

# 
plt.xlabel("years of experience")
plt.ylabel("salary")

plt.show()

