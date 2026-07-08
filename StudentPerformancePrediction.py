import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
data=pd.read_csv('student_performance.csv')
X=data[['StudyHours','Attendance','PreviousMarks','AssignmentScore']]
y=data['FinalMarks']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
model=LinearRegression().fit(X_train,y_train)
pred=model.predict(X_test)
print('R2:',r2_score(y_test,pred))
print(model.predict([[8,92,80,85]]))
