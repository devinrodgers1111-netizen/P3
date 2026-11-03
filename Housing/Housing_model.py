#you will implement linear regression with multiple variables to
#predict the prices of houses. Suppose you are selling your house and you
#want to know what a good market price would be. One way to do this is to
#first collect information on recent houses sold and make a model of housing
#prices.
#The file Housing_data.txt contains a training set of housing prices in Portland, Oregon. The first column is the size of the house (in square feet), the
#second column is the number of bedrooms, and the third column is the price of the house.
#
#  Instructions
#  ------------
#
#  This file will give you outlines to get started on buiding a linear regression model.
#

#personal note: Open the folder with both files so that it works

## ================ Part 1: Feature Normalization ================

## Load Housing_data.txt
# ====================== YOUR CODE HERE ======================
from statistics import median_grouped
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv(                          #reads housing data file 
    "Housing_data.txt",                      #data file name
    header=None,                             #no header for file
    names=["size", "bedrooms", "price"]      #three colum names
)

X = data[["size", "bedrooms"]].values        #stores house size and bedroom count as input features
y = data["price"].values                     #stores house prices as the values for prediction
m = len(y)                                   #counts number of houses in dataset
####################################################################

# Print out some data points
# ====================== YOUR CODE HERE ======================
print("First Five Training")                 #prints label for dataset
print(data.head())                           #prints the first five rows of dataset

####################################################################


# Scale features and set them to zero mean
# ====================== YOUR CODE HERE ======================
#               First, for each feature dimension, compute the mean
#               of the feature and subtract it from the dataset.  
#               Next, compute the standard deviation of each feature and divide
#               Next, You need to perform the normalization separately for each feature. 

mean = np.mean(X, axis=0)                   #finds the average size and average bedroom count
sigma = np.std(X, axis=0)                   #finds the standard deviation of each feature

X_norm = (X - mean) / sigma                 #scales each feature with mean and standard deviation

X_norm = np.c_[np.ones(m), X_norm]          #adds a column of ones for the intercept

####################################################################

# Print out some data points
# ====================== YOUR CODE HERE ======================
print("\nFirst Five Normalized")            #prints a label for the normalized data
print(X_norm[:5])                           #prints the first five rows of normalized features

####################################################################

## =================== Part 2: Cost and Gradient descent ===================

# ====================== YOUR CODE HERE ======================
def compute_cost(X, y, theta):                  #defines a function that measures prediction error
    predictions = X @ theta                     #uses the model to predict prices
    errors = predictions - y                    #finds each difference from prediction to find the real price
    return np.sum(errors ** 2) / (2 * len(y))   #returns the average squared error and is divided by two

# Some gradient descent settings
iterations = 1500;                              #sets how many times the model updates
alpha = 0.01;                                   #sets the learning rate/size of update

# Init Theta and Run Gradient Descent 

theta = np.zeros(X_norm.shape[1])              #starts each parameter at zero
J_history = []                                 #list to store the cost after each update

for iter in range(iterations):                 #repeats the training steps

    # Perform a single gradient step on the parameter thetas. 

    predictions = X_norm @ theta               #predicts prices using the current parameters
    errors = predictions - y                   #finds the errors in those predictions
    gradient = (X_norm.T @ errors) / m         #calculates the gradient for each parameter
    theta = theta - alpha * gradient           #updates the parameters and reduces the cost

    #   After updating theta, compute the cost of using the current values of theta to fit the data points 
    #   You should set J_history(iter) to the cost of using the current values of theta. 
    #   Save the cost J in every iteration
        
    J = compute_cost(X_norm, y, theta)         #calculates the cost using the updated parameters
    J_history.append(J)                        #saves this iterations cost in the list

####################################################################

# Plot the convergence graph
# ====================== YOUR CODE HERE ======================
plt.plot(range(iterations), J_history)        #plots the cost for each iteration
plt.xlabel("Iterations")                      #labels horizontal axis
plt.ylabel("Cost J")                          #labels vertical axis
plt.title("Gradient Descent Convergence")     #graph title
plt.show()                                    #displays the graph

####################################################################

# Display gradient descent's result
# ====================== YOUR CODE HERE ======================
print("\nFinal theta values:")               #prints a label for the final model parameters
print(theta)                                 #prints the learned parameter values

print("\nFinal cost:")                       #prints a label for final cost
print(J_history[-1])                         #prints the last cost saved during training

####################################################################

# Estimate the price of a 1650 sq-ft, 3 br house
# ====================== YOUR CODE HERE ======================
house = np.array([[1650, 3]])               #stores the size and bedroom count of the house to predict
house_norm = (house - mean) / sigma         #scales the new house using the training data's mean and standard deviation
house_norm = np.c_[np.ones(1), house_norm]  #adds the intercept column to the new house's features

predicted_price = house_norm @ theta        #predicts the house price using the trained parameters

print("\nPrice prediction - 1650 sq-ft, 3-bedroom house:")       #prints a label for the prediction
print(f"${predicted_price[0]:,.2f}")                             #prints the predicted price with "$" and two decimal places

####################################################################