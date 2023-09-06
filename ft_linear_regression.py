import pandas as pd
import numpy as np
from Linear_regression import MyLinearRegression as MyLR
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sys

def find_best_params(X_train, Y_train):
	#Hyperparameter tuning method to find the best thetas to use
	theta0_values = np.linspace(np.min(X_train), np.max(X_train), 11)
	theta1_values = np.linspace(np.min(Y_train), np.max(Y_train), 11)
	best_loss = np.inf
	best_theta0 = None
	best_theta1 = None

	for theta0 in theta0_values:
		for theta1 in theta1_values:
			theta = np.array([[theta0], [theta1]])
			linear_model = MyLR(theta, alpha=0.01, max_iter =1000)
			linear_model.fit_(X_train, Y_train, 0)
			Y_pred = linear_model.predict_(X_train)
			loss = linear_model.loss_(Y_train, Y_pred)
			if loss < best_loss:
				best_loss = loss
				best_theta0 = theta0
				best_theta1 = theta1
			
	return np.array([[best_theta0], [best_theta1]])

def normalize(data): #normalizing the data and scaling to the range of 0-1
	mean = np.mean(data)
	std = np.std(data)
	new_norm = (data - mean) / std
	new_scaled = (new_norm  - new_norm.min() ) / (new_norm.max() - new_norm.min())
	
	return (new_scaled)

def main():
	if (len(sys.argv) != 2):
		print("correct usage of the program: python scripy.py data.csv")
		exit (1)
	
	#read the data
	data = pd.read_csv(sys.argv[1])
	X_km = np.array(data['km']).reshape(-1, 1)
	Y_price = np.array(data['price']).reshape(-1, 1)

	#normalize the data to the range of 0-1
	X_km_scaled = normalize(X_km)
	Y_price_scaled = normalize(Y_price)

	#visualize the data
	plt.scatter(X_km_scaled, Y_price_scaled, color='deepskyblue')
	# plt1.scatter(X_km, Y_price, color='blue')

	# Split the data into training, and test sets
	X_train, X_test, Y_train, Y_test = train_test_split(X_km_scaled, Y_price_scaled, test_size=0.3, random_state=42, shuffle=True)

	# find the best thetas to start with
	best_thetas = find_best_params(X_train, Y_train)
	# best_thetas = np.array([[0.7491029520822106], [0.16163793103448273]])#for development purpose we just replace the best thetas

	linear_model_best = MyLR(best_thetas, alpha=0.01, max_iter =10000)
	linear_model_best.fit_(X_train, Y_train, 1)

	#predict using the test set & evaluate
	Y_pred = linear_model_best.predict_(X_test)
	MSE = linear_model_best.mse_(Y_test, Y_pred)
	MAE = linear_model_best.mae_(Y_test, Y_pred)

	# plot the model
	plt.plot(X_test, Y_pred, color='limegreen')
	plt.xlabel("mileage (in km)")
	plt.ylabel("price (in euro)")
	plt.title("Price = f(Mileage) | Normalized")
	plt.text(0.95, 0.95, "MSE: {:.2f}, MAE: {:.2f}".format(MSE, MAE), ha='right', va='top', transform=plt.gca().transAxes)

 	# plot the cost
	plt.figure()
	plt.plot(range(len(linear_model_best.losses)), linear_model_best.losses, color='deepskyblue')
	plt.xlabel("Number of iterations")
	plt.ylabel("Cost")
	plt.title("Cost = f(iteration) | L.rate = 0.01")

	plt.show()

if __name__ == "__main__":
	main()
