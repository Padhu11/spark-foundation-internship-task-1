# spark-foundation-internship-task-1
Loading…
task 1.ipynb
task 1.ipynb_
PADMAJA.G

The Sparks Foundation
Data Science and Business Analytics Internship

Task 1:Linear Regression

This simple Linear Regression task focuses on predicting the score obtained by student based on the number of study hours.

importing libraries
[ ]
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  
%matplotlib inline
Reading data
[ ]
url = "http://bit.ly/w-data"
data =pd.read_csv(url)
print("Data imported successfully")
data.head(12)

[ ]
# check the info of data
data.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 25 entries, 0 to 24
Data columns (total 2 columns):
 #   Column  Non-Null Count  Dtype  
---  ------  --------------  -----  
 0   Hours   25 non-null     float64
 1   Scores  25 non-null     int64  
dtypes: float64(1), int64(1)
memory usage: 528.0 bytes
plotting the distribution of scores
[ ]
plt.title("Raw data - Hours studied vs Marks obtained")
plt.xlabel("number of study hours")
plt.ylabel("Marks obtained")
plt.scatter(data.Hours,data.Scores,color='blue',label='Data Distribution')
plt.legend(['Data Distribution'])

we can clearly see that there is a positive linear realtion between number of study hours and marks obtained.

Training the model
[ ]
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
[ ]
#Splitting training and testing data
x=data.iloc[:,:-1].values
y=data.iloc[:,1].values
x_train, x_test, y_train, y_test= train_test_split(x, y,train_size=0.80,test_size=0.20,random_state=0)
[ ]
from sklearn.linear_model import LinearRegression
linearRegressor= LinearRegression()
linearRegressor.fit(x_train, y_train)
y_predict= linearRegressor.predict(x_train)
Traning the Algorithm
[ ]
 regressor = LinearRegression()  
regressor.fit(x_train, y_train)  
print("Training complete.")
Training complete.
[ ]
# Plotting the regression line
line = regressor.coef_*x+regressor.intercept_
# Plotting for the test data
plt.scatter(x, y,color='green')
plt.plot(x, line);
plt.show()

Checking the accuracy scores for Training and test set
[ ]
print('Test Score')
print(regressor.score(x_test, y_test))
print('Training Score')
print(regressor.score(x_train, y_train))
Test Score
0.9454906892105354
Training Score
0.9515510725211553
predictions
[ ]
y_test
array([20, 27, 69, 30, 62])
[ ]
y_predict
array([39.67865467, 20.84840735, 79.32128059, 70.40168976, 12.91988217,
       52.56250809, 78.33021494, 34.72332643, 84.27660883, 93.19619966,
       62.47316457, 36.70545772, 28.77693254, 56.52677068, 28.77693254,
       86.25874013, 26.79480124, 49.58931115, 90.22300272, 46.6161142 ])
[ ]
y_predict[:5]
array([39.67865467, 20.84840735, 79.32128059, 70.40168976, 12.91988217])
[ ]
data= pd.DataFrame({'Actual': y_test,'Predicted': y_predict[:5]})
data

[ ]

#Let's predict the score for 9.25 hpurs
print('Score of student who studied for 9.25 hours a dat', regressor.predict([[9.25]]))
Score of student who studied for 9.25 hours a dat [93.69173249]
Model Evaluation
[ ]
#Checking the efficiency 
mean_squ_error = mean_squared_error(y_test, y_predict[:5])
mean_abs_error = mean_absolute_error(y_test, y_predict[:5])
print("Mean Squred Error:",mean_squ_error)
print("Mean absolute Error:",mean_abs_error)
Mean Squred Error: 914.5549752244242
Mean absolute Error: 25.126667098277874

