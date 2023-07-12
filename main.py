import pandas as pd
import numpy as np
from Linear_regression import MyLinearRegression as MyLR
import matplotlib.pyplot as plt
import math
from TinyStatistician import TinyStatistician as TS
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def main():
	#read the data
	data = pd.read_csv("Data.csv")
	X_km = np.array(data['km']).reshape(-1, 1)
	Y_price = np.array(data['price']).reshape(-1, 1)

	#normalize the data
	scaler_X = StandardScaler()
	X_km_scaled = scaler_X.fit_transform(X_km)
	
	scaler_Y = StandardScaler()
	Y_price_scaled = scaler_Y.fit_transform(Y_price)

	#visualize the data
	plt.scatter(X_km_scaled, Y_price_scaled, color='deepskyblue')
	
	# Split the data into training, and test sets
	X_train, X_test, Y_train, Y_test = train_test_split(X_km_scaled, Y_price_scaled, test_size=0.3, random_state=42)

	#Hyperparameter tuning
	theta0_values = np.linspace(np.min(X_train), np.max(X_train), 20)
	theta1_values = np.linspace(np.min(Y_train), np.max(Y_train), 20)
	best_loss = np.inf
	best_theta0 = np.nan
	best_theta1 = np.nan
	losses = []

	for theta0 in theta0_values:
		for theta1 in theta1_values:
			theta = np.array([[theta0], [theta1]])
			linear_model = MyLR(theta, alpha=0.001, max_iter =7000)
			linear_model.fit_(X_train, Y_train)
			Y_pred = linear_model.predict_(X_train)
			loss = linear_model.loss_(Y_train, Y_pred)
			print(loss, " ,(theta0, theta1): ", theta0, ", ", theta1)
			losses.append(loss)
			if loss < best_loss:
				best_loss = loss
				best_theta0 = theta0
				best_theta1 = theta1
		
	# Train the model with the best thetas
	best_thetas = np.array([[best_theta0], [best_theta1]])
	linear_model_best = MyLR(best_thetas, alpha=0.001, max_iter =7000)
	linear_model_best.fit_(X_test, Y_test)
	Y_pred = linear_model_best.predict_(X_test)
	MSE = linear_model.mse_(Y_test, Y_pred)
	MAE = linear_model.mae_(Y_test, Y_pred)

	# plot the model
	plt.plot(X_test, Y_pred, color='limegreen')
	plt.xlabel("mileage (in km)")
	plt.ylabel("price (in euro)")
	plt.legend(loc='lower right')
	plt.title("Price = f(Mileage) | Normalized")
	plt.text(0.95, 0.95, "MSE: {:.2f} MAE: {:.2f}".format(MSE, MAE), ha='right', va='top', transform=plt.gca().transAxes)

 	# plot the loss versus number of iterations
	plt.figure()
	plt.plot(range(len(losses)), losses, color='deepskyblue')
	plt.xlabel("Number of iterations")
	plt.ylabel("Cost")
	plt.title("Cost = f(iteration) | L.rate = 0.01")

	plt.show()

if __name__ == "__main__":
	main()
