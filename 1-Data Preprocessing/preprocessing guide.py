
# Data Preprocessing

# Importing the libraries
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('C:\Machine Learning A-Z\Part 1 - Data Preprocessing\Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

# -----------------------------------------------------
# جایگزینی داده های بدون مقدار
from sklearn.preprocessing import Imputer

# ایجاد یک آبجت برای استفاده
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)

# استفاده از آبجکت برای محاسبه بر روی ستون های مورد نظر
imputer = imputer.fit(X[:, 1:3])

# جایگزینی موارد محاسبه شده در ماتریس اصلی
X[:, 1:3] = imputer.transform(X[:, 1:3])

# -----------------------------------------------------
# تبدیل داده های کیفی به کمی
from sklearn.preprocessing import LabelEncoder

# برای متغیر مستقل
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])

# برای متغیر وابسته
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# تغییر و تفکیک داده های یک ستون به صفر و یک
from sklearn.preprocessing import OneHotEncoder

onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()

# چون داده وابسته تنها تبدیل به صفر و یک شد نیازی به تبدیل آن نیست

# -----------------------------------------------------
# تقسیم کردن دیتا به دو قسمت ست آماده سازی و ست آزمون

"""
 در الگوریتم های ماشین لرنینگ داده ها را برای تحلیل و برآورد به دو بخش
 آماده سازی و آزمون تقسیم میکنیم
 در بخش آماده سازی داده ها تحلیل شده و برآورد محاسبه می شود
 در بخش آزمون مقادیر برآوردشده با واقعی برای داده های آزمون مقایسه می شوند
 تا میزان قدرت تحلیل بررسی شود

"""

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# یکسان سازی مقیاس داده ها
from sklearn.preprocessing import StandardScaler

#  تغییر مقیاس داده مستقل
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)

