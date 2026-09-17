import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

# reading dataset
dataset = pd.read_csv('veriseti.csv')

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# for lineer regresyon model train
lineer_regresyon_model = LinearRegression()
lineer_regresyon_model.fit(x,y)

#We generate prediction values based on the dataset set aside for testing.

predictyear = [[18],[19],[20],[21]]
predictresult = lineer_regresyon_model.predict(predictyear)

print(str(predictresult) + " forecast results for : " + str(predictyear))

# Visualize the test data as a graph.
plt.scatter(x, y, color= 'blue')
plt.scatter(predictyear, predictresult, color = 'red')
plt.plot(x, lineer_regresyon_model.predict(x),color = 'blue')
plt.title("Emission Estimation")
plt.xlabel("Year")
plt.ylabel("Emissions")
plt.show()
