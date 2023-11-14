import pandas as pd
import numpy as np
from Linear_regression import MyLinearRegression as MyLR
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sys
import json

def find_best_params(X_train, Y_train, _alpha, _iter):
	#Hyperparameter tuning method to find the best thetas to use
	theta0_values = np.linspace(np.min(X_train), np.max(X_train), 11)
	theta1_values = np.linspace(np.min(Y_train), np.max(Y_train), 11)
	best_loss = np.inf
	best_theta0 = None
	best_theta1 = None

	for theta0 in theta0_values:
		for theta1 in theta1_values:
			theta = np.array([[theta0], [theta1]])
			linear_model = MyLR(theta, alpha=_alpha, max_iter =_iter)
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

def train_test(best_thetas, X, Y, _alpha, _iter):
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42, shuffle=True)

	linear_model_best = MyLR(best_thetas, alpha=_alpha, max_iter =_iter)
	linear_model_best.fit_(X_train, Y_train, 1)

	Y_pred = linear_model_best.predict_(X_test)
	MSE = linear_model_best.mse_(Y_test, Y_pred)
	MAE = linear_model_best.mae_(Y_test, Y_pred)

	fig, axes = plt.subplots(3, 1, figsize=(6, 12))

	# plot the distribution of the original data
	axes[0].hist(X, color='blue', edgecolor='black')
	axes[0].set_title("Normalized Data Distribution")
	axes[0].set_xlabel("Mileage (in km)")
	axes[0].set_ylabel("Frequency")
	
	# plot the model
	axes[1].scatter(X, Y, color='deepskyblue')
	axes[1].plot(X_test, Y_pred, color='limegreen')
	axes[1].set_xlabel("mileage (in km)")
	axes[1].set_ylabel("price (in euro)")
	axes[1].set_title("Price = f(Mileage) | Normalized")
	axes[1].text(0.95, 0.95, "MSE: {:.2f}, MAE: {:.2f}".format(MSE, MAE), ha='right', va='top', transform=plt.gca().transAxes)

	# plot the cost
	axes[2].plot(range(len(linear_model_best.losses)), linear_model_best.losses, color='deepskyblue')
	axes[2].set_xlabel("Number of iterations")
	axes[2].set_ylabel("Cost")
	axes[2].set_title("Cost = f(iteration) | L.rate = 0.01")

	plt.tight_layout()
	plt.savefig("output.png")

def main():
	if (len(sys.argv) < 4):
		print("correct usage of the program: python scripy.py data.csv iter lr mode")
		exit (1)
	
	data = pd.read_csv(sys.argv[1])
	X_km = np.array(data['km']).reshape(-1, 1)
	Y_price = np.array(data['price']).reshape(-1, 1)
	X_km_scaled = normalize(X_km)
	Y_price_scaled = normalize(Y_price)

	if (sys.argv[4] == '1'): #finding best params
		best_thetas = find_best_params(X_train, Y_train, float(sys.argv[3]), int(sys.argv[2]))
		with open('best_params.json', 'w') as file:
			json.dump(best_thetas.tolist(), file)
	elif (sys.argv[4] == '2'): #training mode
		with open('best_params.json', 'r') as file:
			best_thetas = np.array(json.load(file))
		train_test(best_thetas, X_km_scaled, Y_price_scaled, float(sys.argv[3]), int(sys.argv[2]))
		

if __name__ == "__main__":
	main()
