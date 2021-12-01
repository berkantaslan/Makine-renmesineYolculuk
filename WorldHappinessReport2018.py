# Importing Libraries
import numpy as np
import pandas as pd

# Taking Data
data = pd.read_csv('WorldHappinessReport-2018.csv')

# Data Imputer
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='constant',fill_value= 0)
impdata = data.iloc[:,2:9].values
imputer = imputer.fit(impdata[:,2:9])
impdata[:,2:9] = imputer.transform(impdata[:,2:9])
data.iloc[:,2:9] = impdata[:,:]

# Data Exploratory
print(data.columns)
print(data.info())
print(data.describe())
df = data["Country or region"]
dff = data.value_counts()

# Selecting Columns x and y
x = data.iloc[:,3:].values
y = data.iloc[:,2:3].values

# Splitting as Test and Train
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)

# Scaling of Data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)
Y_train = sc.fit_transform(y_train)
Y_test = sc.fit_transform(y_test)

# Building Model
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train, y_train)
print("b0: ", lr.intercept_)
print("other b: ", lr.coef_)

# Predicting
y_pred = lr.predict(x_test)
prediction = lr.predict(np.array([[1.148, 1.38, 0.686, 0.324, 0.106, 0.109]]))
print("Prediction is ", prediction)

# Controlling of Success of Data and Model
import statsmodels.regression.linear_model as sm
X = np.append(arr = np.ones((156,1)).astype(int), values=x, axis=1)
r_ols = sm.OLS(endog = y, exog = X)
r = r_ols.fit()
print(r.summary())
