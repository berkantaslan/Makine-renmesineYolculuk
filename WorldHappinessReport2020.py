# Importing Libraries
import numpy as np
import pandas as pd

# Taking Data
data = pd.read_csv('WorldHappinessReport-2020.csv')

# Data Impute
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='constant',fill_value= 0)
impdata = data.iloc[:,4:20].values
imputer = imputer.fit(impdata[:,4:20])
impdata[:,4:20] = imputer.transform(impdata[:,4:20])
data.iloc[:,4:20] = impdata[:,:]

# Data Eploratory
print(data.columns)
print(data.info())
print(data.describe())
df = data["Country name"]
dff = data.value_counts()

# Selecting Columns x and y
x = data.iloc[:,4:].values
y = data.iloc[:,2:3].values

# Splitting as Test and Train
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)

# Scaling of Data-Standardization
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

# Controlling of Success of Data and Model
import statsmodels.regression.linear_model as sm
X = np.append(arr = np.ones((153,1)).astype(int), values=x, axis=1)
r_ols = sm.OLS(endog = y, exog = X)
r = r_ols.fit()
print(r.summary())
