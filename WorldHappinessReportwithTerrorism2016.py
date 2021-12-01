# Importing Libraries
!pip install bubbly

import numpy as np
import pandas as pd   
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.gridspec as grid_spec
import seaborn as sns
import plotly.graph_objs as go
import plotly.offline as po
from bubbly.bubbly import bubbleplot
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected = True)

# Taking Data
data = pd.read_csv('WorldHappinessReportwithTerrorism-2016.csv')

# Data Eploratory
print(data.columns)
print(data.info())
print(data.describe())

# Selecting Columns x and y
x = data.iloc[:,6:].values
y = data.iloc[:,3:4].values

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

# Visualization
figure = bubbleplot(dataset = data2, x_column = 'happinessscore', y_column = 'healthlifeexpectancy', 
    bubble_column = 'country', size_column = 'economysituation', color_column = 'region', 
    x_title = "Happiness Score", y_title = "Health Life Expectancy", title = 'Happiness vs Health Life Expectancy vs Economy',
    x_logscale = False, scale_bubble = 1, height = 650)

po.iplot(figure)

# Controlling of Success of Data and Model
import statsmodels.regression.linear_model as sm
X = np.append(arr = np.ones((157,1)).astype(int), values=x, axis=1)
r_ols = sm.OLS(endog = y, exog = X)
r = r_ols.fit()
print(r.summary())
