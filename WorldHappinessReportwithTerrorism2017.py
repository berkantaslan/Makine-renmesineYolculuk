# Importing Libraries
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
data = pd.read_csv('WorldHappinessReportwithTerrorism-2017.csv')

# Data Eploratory
print(data.columns)
print(data.info())
print(data.describe())

# Selecting Columns x and y
x = data.iloc[:,5:].values
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

# Visualization
trace1 = [go.Choropleth(
               colorscale = 'Electric',
               locationmode = 'country names',
               locations = data3['country'],
               text = data3['country'], 
               z = data3['happinessrank'],
               )]

layout = dict(title = 'Happiness Rank World',
                  geo = dict(
                      showframe = True,
                      showocean = True,
                      showlakes = True,
                      showcoastlines = True,
                      projection = dict(
                          type = 'hammer'
        )))


projections = [ "equirectangular", "mercator", "orthographic", "natural earth","kavrayskiy7", 
               "miller", "robinson", "eckert4", "azimuthal equal area","azimuthal equidistant", 
               "conic equal area", "conic conformal", "conic equidistant", "gnomonic", "stereographic", 
               "mollweide", "hammer", "transverse mercator", "albers usa", "winkel tripel" ]

buttons = [dict(args = ['geo.projection.type', y],
           label = y, method = 'relayout') for y in projections]

annot = list([ dict( x=0.1, y=0.8, text='Projection', yanchor='bottom', 
                    xref='paper', xanchor='right', showarrow=False )])


# Update Layout Object
layout[ 'updatemenus' ] = list([ dict( x=0.1, y=0.8, buttons=buttons, yanchor='top' )])
layout[ 'annotations' ] = annot


fig = go.Figure(data = trace1, layout = layout)
po.iplot(fig)

# Controlling of Success of Data and Model
import statsmodels.regression.linear_model as sm
X = np.append(arr = np.ones((155,1)).astype(int), values=x, axis=1)
r_ols = sm.OLS(endog = y, exog = X)
r = r_ols.fit()
print(r.summary())
