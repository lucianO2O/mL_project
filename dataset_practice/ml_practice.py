import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

# read the csv file
irisData = pd.read_csv("iris_dataset_with_class_information.csv")
# print all the columns
print(irisData)

# print column names
print(irisData.columns)

# change the names of the columns
irisData = irisData.rename(columns={'sepal.length.in.cm': 'sepal_length', 'sepal.width.in.cm': 'sepal_width', 'petal.length.in.cm':'petal_length', 'petal.width.in.cm': 'petal_width'})
print(irisData)

# print (number of rows, number of columns)
print(irisData.shape)

# view first 5 rows
print(irisData.head())

# view last 5 rows
print(irisData.tail())

# to pick apart some rows and columns, subsetting
#   when defining range:
#       - m:n means going up to but not including n
#       - m: means m -> end
#       - :m means 0 -> m, not including m
# ex.
print(irisData.iloc[0:10, 0:5]) # first = range of rows, second = range of columns
print(irisData.iloc[:10, :5]) # does same thing as method above
print(irisData.iloc[145:, 2:]) # prints last 5 rows under lst 3 columns

# look at specific columns
print(irisData['sepal_length'])

# can list multiple specific column names by placing in variable
sepal_LAndW = ['sepal_length', 'sepal_width']
# will only print the 2 columns above
print(irisData[sepal_LAndW])

# filter certain rows with a particular column name
#   - this will find all the samples with species 'versicolor'
irisData_versicolor = irisData.loc[irisData['species'] == 'versicolor']
# print number of rows + columns under that species
print(irisData_versicolor.shape)

# shuffling the dataset before use
irisData_randomized = irisData.sample(frac=1, random_state=42) # random_state=42 ensures that shuffling is reproducible, if you use this same state again, it will shuffle in the same way, kind of like a game seed
#                                      ^- specifies that you want to sample all rows
irisData_randomized = irisData_randomized.reset_index(drop=True)
# all pandas dataframe have indexes associated with each rows. After shuffling, the original indexes remain in each row, so this function ensures a new index is created and old ones are dropped
print(irisData_randomized)

X = irisData_randomized.iloc[:,0:4] # data without the class labels, all numeric data
y = irisData_randomized.iloc[:,4] # the class labels/ outcome variable of training data, the flower species
X.head() # checking the features
y.head() # checking y, these verify the split

# splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=12, shuffle=True)
# test_size=0.20 → 20% of data goes into testing, 80% into training
# random_state=12 → ensures reproducibility (you get the same split every time)
# shuffle=True → shuffles rows before splitting (to avoid any order bias)
print(X_train.shape)
# gives dimensions of training set feature, i.e. (120, 4)