# DataScience x linear_regression | 42Paris
## Objective
To create a linear regression function, train the model on the given dataset, save generated indexes and use 
them to predict car price depending on it's mileage.

## Usage
git clone https://github.com/shimazadeh/Ft_linear_regression.git Linear-regression
cd Linear-regression
python3 ft_linear_regression.py [path/to/dataset.csv]

## Approach
Data Preprocessing: The program reads the dataset from the CSV file, normalizes the data to the range of 0-1.
Train-Test Splitting: The program splits the dataset into training and test sets.
Hyperparameter Tuning: It uses hyperparameter tuning method to find the best initial parameters (thetas) for the linear regression model.
Model Training: The linear regression model is trained using gradient descent. The program provides options to visualize the training process and print the model parameters, Loss, MSE and MAE each iteration.
Model Evaluation: After training, the program predicts prices using the test set and calculates the Mean Squared Error (MSE) and Mean Absolute Error (MAE) to evaluate the model's performance.
Visualization: The program visualizes the normalized dataset, the regression model, and the cost function as shown below: 
![Figure_1](https://github.com/shimazadeh/Ft_linear_regression/assets/67879533/2b4d502c-d9a8-4c9c-a3b0-f62ac38bd210)
![Figure_2](https://github.com/shimazadeh/Ft_linear_regression/assets/67879533/2481e403-d1b6-40cc-9aa0-6ac0ad6ed66e)
