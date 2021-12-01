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
data = pd.read_csv('WorldHappinessReportwithTerrorism-2015.csv')

# Data Exploratory
print(data.columns)
print(data.info())
print(data.describe())

# Selecting Columns x and y
x = data.iloc[:,5:].values
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

# Predicting
y_pred = lr.predict(x_test)
prediction = lr.predict(np.array([[1.198274,1.337753,0.637606,0.300741,0.099672,0.046693,1.879278,181]]))
print("Prediction is ", prediction)

y_pred = lr.predict(x_test)
prediction = lr.predict(np.array([[1.198274,1.337753,0.637606,0.300741,0.099672,0.046693,1.879278,181]]))
print("Prediction is ", prediction)

# Visualization
low_c = '#dd4124'
high_c = '#009473'
background_color = '#fbfbfb'
fig = plt.figure(figsize=(12, 10), dpi=150,facecolor=background_color)
gs = fig.add_gridspec(3, 3)
gs.update(wspace=0.2, hspace=0.5)

newdata1 = data1.iloc[:,4:]
categorical = [var for var in newdata1.columns if newdata1[var].dtype=='O']
continuous = [var for var in newdata1.columns if newdata1[var].dtype!='O']

happiness_mean = data1['happinessscore'].mean()

data1['lower_happy'] = data1['happinessscore'].apply(lambda x: 0 if x < happiness_mean else 1)

plot = 0
for row in range(0, 3):
    for col in range(0, 3):
        locals()["ax"+str(plot)] = fig.add_subplot(gs[row, col])
        locals()["ax"+str(plot)].set_facecolor(background_color)
        locals()["ax"+str(plot)].tick_params(axis='y', left=False)
        locals()["ax"+str(plot)].get_yaxis().set_visible(False)
        locals()["ax"+str(plot)].set_axisbelow(True)
        for s in ["top","right","left"]:
            locals()["ax"+str(plot)].spines[s].set_visible(False)
        plot += 1

plot = 0

Yes = data1[data1['lower_happy'] == 1]
No = data1[data1['lower_happy'] == 0]

for variable in continuous:
        sns.kdeplot(Yes[variable],ax=locals()["ax"+str(plot)], color=high_c,ec='black', shade=True, linewidth=1.5, alpha=0.9, zorder=3, legend=False)
        sns.kdeplot(No[variable],ax=locals()["ax"+str(plot)], color=low_c, shade=True, ec='black',linewidth=1.5, alpha=0.9, zorder=3, legend=False)
        locals()["ax"+str(plot)].grid(which='major', axis='x', zorder=0, color='gray', linestyle=':', dashes=(1,5))
        locals()["ax"+str(plot)].set_xlabel(variable, fontfamily='monospace')
        plot += 1
        
Xstart, Xend = ax0.get_xlim()
Ystart, Yend = ax0.get_ylim()

ax0.text(Xstart, Yend+(Yend*0.5), 'Differences Between Happy & Unhappy Countries', fontsize=15, fontweight='bold', fontfamily='sansserif',color='#323232')
ax0.text(Xstart, Yend+(Yend*0.25), 'There are large differences, with GDP & Social Support being clear perhaps more interesting though,unhappy\ncountries appear to be more generous.', fontsize=10, fontweight='light', fontfamily='monospace',color='gray')

plt.show()


# Controlling of Success of Data and Model
import statsmodels.regression.linear_model as sm
X = np.append(arr = np.ones((158,1)).astype(int), values=x, axis=1)
r_ols = sm.OLS(endog = y, exog = X)
r = r_ols.fit()
print(r.summary())
