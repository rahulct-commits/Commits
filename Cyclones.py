import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.cov_struct import (Exchangeable,
    Independence,Autoregressive)
from statsmodels.genmod.families import Poisson

dataset = pd.read_csv('D:\\Python Scripts\\Cyclones.csv')


### EDA ###

dataset.head()
dataset.columns
dataset.describe()


# No NAs
dataset.isna().any()

## Histograms

dataset2 = dataset.drop(columns = ['year','TC'])

fig = plt.figure(figsize=(15, 12))
plt.suptitle('Histograms of Numerical Columns', fontsize=20)
for i in range(dataset2.shape[1]):
    plt.subplot(6, 3, i + 1)
    f = plt.gca()
    f.set_title(dataset2.columns.values[i])

    vals = np.size(dataset2.iloc[:, i].unique())
    if vals >= 100:
        vals = 100
    
    plt.hist(dataset2.iloc[:, i], bins=vals, color='#3F5D7D')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

#Correlation with Response Variable
dataset2.corrwith(dataset.TC).plot.bar(
        figsize = (20, 10), title = "Correlation with TC", fontsize = 15,
        rot = 45, grid = True)

X=['MJJ','JJA','JAS']	
# building the model

fam = Poisson()
ind = Independence()
model1 = GEE.from_formula("TC ~ MJJ + JJA + JAS", cov_struct=ind, family=fam)
result1 = model1.fit()
print(result1.summary())

# testing the model

predVals = poisson_res.predict(X)

plt.plot(range(len(TC)), TC, 'r*-', range(len(TC)), predVals, 'bo-')
plt.title('Train dataset Real vs. Predicted Values')
plt.legend(['Real Values', 'Predicted Values'])
plt.show()