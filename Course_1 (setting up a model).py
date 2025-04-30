import pandas as pd


## Save File path top variable for easy access
melbourne_file_path = (
    r"C:\Users\Hen Baynie\Python Workspace\Machine Learning Python Files/melb_data.csv"
)

## Read the data and store it in DataFrame
melbourne_data = pd.read_csv(melbourne_file_path)

## Print summary of data
# print(melbourne_data.describe())

## See all the columns in the data
# print(melbourne_data.columns)

## Dropping the empty data from the data set
# melbourne_data = melbourne_data.dropna(axis=0)

## Prediction Target: Using the . to select a specific column and assign to a variable this is called the Prediction Target
y = melbourne_data.Price

## Features: Select columns using a list that will be used to predict the prediction target
melbourne_features = ["Rooms", "Bathroom", "Landsize", "Lattitude", "Longtitude"]
X = melbourne_data[melbourne_features]

## Use the head() method to grab just the first 5 rows to check the dataset
# print(X.head())

## Building a Model: Uses Sklearn, importing the model type
from sklearn.tree import DecisionTreeRegressor

## Defining the model (Best practice is to assign a random_state)
melbourne_model = DecisionTreeRegressor(random_state=1)
## Fit model using our Features and Prediction Target
melbourne_model.fit(X, y)
## Print model
# print("Making predictions for the following 5 houses:")
# print(X.head())
# print("The predictions are: ")
# print(melbourne_model.predict(X.head()))

## Calculating the Mean Absolute Error in our model (error = actual - prediction)
from sklearn.metrics import mean_absolute_error

# predicted_home_prices = melbourne_model.predict(X)
# print(mean_absolute_error(y, predicted_home_prices))

## Split the data into 2 sections, one to train the model, one to validate the models accuracy. This ensures the model will be accurate with new data
from sklearn.model_selection import train_test_split

## The split is based on RNG, apply a number to the random_state argument to guarentee we get the same split everytime we run the script
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)
## Define new model with the data split
# new_melbourne_model = DecisionTreeRegressor(random_state=1)
## Fit new model
# new_melbourne_model.fit(train_X, train_y)

## Use the validation data to test for accuracy on the new model using *new* data
# val_predictions = new_melbourne_model.predict(val_X)
# print(mean_absolute_error(val_y, val_predictions))


## Creating a function to get MAE with max leaf nodes to find a sweet spot between overfitting and underfitting
# def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
#     model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
#     model.fit(train_X, train_y)
#     preds_val = model.predict(val_X)
#     mae = mean_absolute_error(val_y, preds_val)
#     return mae


## Running a for loop to determine best amount of leaves for the decision tree
# for max_leaf_nodes in [5, 50, 500, 5000]:
#     my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
#     print("Max leaf nodes: %d \t \t Mean Absolute Error: %d" % (max_leaf_nodes, my_mae))

## Create a Forest model for more predictive accuracy than a Tree model beacuse it takes multiple trees and averages them out
from sklearn.ensemble import RandomForestRegressor

forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_X, train_y)
melb_preds = forest_model.predict(val_X)
print(mean_absolute_error(val_y, melb_preds))
