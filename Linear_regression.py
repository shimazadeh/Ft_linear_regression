import numpy as np
import math
import pandas as pd

class MyLinearRegression():
	"""
	Description:
	My personnal linear regression class to fit like a boss.lol
	"""
	def __init__(self, thetas, alpha=0.0001, max_iter=1000):
		self.alpha = alpha
		self.max_iter = max_iter
		self.thetas = thetas
		self.losses = []

	@staticmethod
	def add_intercept(x):
		if x is None:
			return 0
		df = pd.DataFrame(x)
		num_columns = len(df.axes[1])
		first_column = df.pop(0)
		df.insert(0, 0, 1)

		i = 1
		while(i < num_columns + 1):
			if (i < num_columns):
				next_column = df.pop(i)
			else:
				next_column = 0
			df.insert(i, i, first_column)
			first_column = next_column
			i = i + 1
		x = df.to_numpy()
		return (x)

	def	gradient(self,x, y, theta): #sum(y(i)_predict - y(i)) / m
		if theta is None or x is None or y is None:
			return 0
		if len(x) != len(y):
			return 0
		x_new = self.add_intercept(x)
		x_t = np.transpose(x_new)
		m = len(x_new)
		J = (np.dot(x_new, theta) - y)
		gradient = np.dot(x_t, J) / m
		return (gradient)

	def fit_(self, x, y):#fits the model to the training dataset
		if x is None or y is None:
			return 0
		if len(x) != len(y):
			return 0
		for i in range(self.max_iter):
			delta_j = self.gradient(x, y, self.thetas)
			if np.isnan(delta_j[0]) or np.isinf(delta_j[1]):
				break
			self.thetas[0] = self.thetas[0] - self.alpha * delta_j[0]
			self.thetas[1] = self.thetas[1] - self.alpha * delta_j[1]
			y_p = self.predict_(x)
			self.losses.append(self.loss_(y, y_p))
		return (self.thetas)

	def predict_(self, x):
		if x is None or self.thetas is None:
			return 0
		z = self.add_intercept(x)
		if (len(pd.DataFrame(z).axes[1]) != len(self.thetas)):
			return 0
		return (np.dot(z, self.thetas))

	def loss_(self, y, y_hat):#using MSE
		MSE = (np.square(np.subtract(y, y_hat)).mean())
		return (MSE)

	@staticmethod
	def mse_(y, y_hat):
		MSE = (np.square(np.subtract(y, y_hat)).mean())
		return (MSE)

	@staticmethod
	def mae_(y, y_hat):
		MAE = np.abs(np.subtract(y, y_hat)).mean()
		return (MAE)