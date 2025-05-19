# My task is to create a simple linear regression model from one of these features(from csv file)
# to predict CO2 emissions of unobserved cars based on that feature

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model

url= "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv"
df = pd.read_csv(url)
cdf = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]

#to see diagrams
# vizualization = cdf[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]
# print(vizualization.hist())

# to see scatter plots of combined fuel consumption and CO2 emission
#plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS, color='blue')
#plt.xlabel('FUELCONSUMPTION_COMB')
#plt.ylabel('EMISSION')
#plt.show()

X = cdf.ENGINESIZE.to_numpy()
Y = cdf.CO2EMISSIONS.to_numpy()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

regressor = linear_model.LinearRegression() # model object
regressor.fit(X_train.reshape(-1, 1), Y_train) # -1 automatically determine the number of rows, 1 feature(single column)
# slope and intercept of the best-fit line
# print("Coef: ", regressor.coef_[0])
# print("Inter: ", regressor.intercept_)
plt.scatter(X_train, Y_train,  color='blue')
plt.plot(X_train, regressor.coef_ * X_train + regressor.intercept_, '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()