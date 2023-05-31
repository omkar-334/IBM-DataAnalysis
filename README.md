# IBM-DataAnalysis    
  
### Week 1 - Importing Datasets    
  
Scientific computing libraries- Numpy, Scipy, Pandas    
DV libraries - Matplotlib, Seaborn    
Algorithmic libraries - Scikit-learn, statsmodels    
  
###### Connection objects- Database connections, manage transactions    
import dbmodule as db    
commit()- commit any pending transaction    
cursor()-returns new cursor using connection    
rollback()- rollback to start of pending transaction    
close()- close database connection    
#create connection and cursor objects    
connection=connect("databasename","username","password")    
cursor=connection.cursor()    
#run queries    
cursor.execute('select * from mytable')    
results=cursor.fetchall()    
#free resources    
cursor.close()    
connection.close()    
Cursor objects - Database queries    
  
### Week 2 - Data Wrangling    
df.dropna()    
axis=0 for rows    
axis=1 for columns    
df.replace(oldvalue,newvalue)    
replace with mean or mode    
df.rename(columns={"city":"City"},inplace=True)    
df["price"]=df["price"].astype("int")    
    
###### Binning  
  
bins=np.linspace(min(df["price"]),max(df["price"]),4)    
groupnames=["low","medium","high"]    
df["bins"]=pd.cut(df["price"],bins,labels=groupnames,include_lowest=True)    
  
###### Categorical Variable to Numeric/Quantitative Data - One-hot Encoding  
  
Assigning 0s and 1s in each category    
pd.get_dummies(df["fuel"])    
  
### Week 3 - Exploratory Data Analysis  
  
##### Methods of normalizing data  
###### 1.Simple Feature Scaling  
Xnew=Xold/Xmax    
  
###### 2.Min-Max  
Xnew=(Xold-Xmin)/Xmax-Xmin    
  
###### 3.Z-score(-3 to +3)  
Xnew=(Xold-average)/SD    
  
##### Methods for Analysis  
###### 1.Descriptive Statistics    
df.descibe() - NaN values are excluded    
  
###### 2.Summarizing categorical Data    
df_counts=df["fuel"].value_counts().to_frame()    
  
###### 3.Groupby for categorical,single or multiple variables    
df_test=df[["drive_wheels",'body_style','price']]    
df_grp=df_test.groupby(['drive_wheels','body_style'],as_index=False).mean()    
  
###### 4.Pivot for one variable along rows and one variable along columns    
df_pivot=df_grp(index="drive-wheels',columns='body-style')    
  
###### 5.Heatmaps    
plt.pcolor(df_pivot)    
plt.colorbar()    
plt.show()    
  
###### 6.Correlation - sns.regplot()  
  
Positive,Negative,Weak,Strong Linear Graphs  
  
###### 7.Correlation Statistics  
  
Correlation Coefficient  
		-1 - Large negative relationship  
		 0 - No relationship  
		+1 - Large Positive relationship  
Pearson Value  
		<0.001 - Strong Certainty  
		<0.05 - Moderate Certainty  
		<0.1 - Weak Certainty  
		>0.1 - No Certainty  
SciPy Library  
p_coef,p_value=stats.pearsonr(df['horsepower'],df['price'])  
  
###### 8. Chi-Square test for association  
  
How likely it is that an observed distribution is due to chance  
Crosstab- table showing the relationship between two or more variables  
  
###### 9. Contingency table - table only shows the relationship between two categorical variables  
  
scipy.stats.chi2_contingency(table,correction=True)  
  
###### 10. Boxplots  
  
![box](https://github.com/omkar-334/IBM-DataAnalysis/assets/40126336/b7228376-e4a0-4e3c-9d03-b31cfe9ba960)  
uq=1.5*IQR above 75th %ile  
lq=1.5*IQR below 25th %ile  
  
###### 11. Chi Square Test  
  
![chi](https://github.com/omkar-334/IBM-DataAnalysis/assets/40126336/9b358320-8de1-4039-948b-1c314f21ac0b)  
Oi-Observed Value  
Ei - Expected Value  
Ei=Rowtotal*Columtotal/Grandtotal  
DoF=(row-1)*(column-1)  
  
### Week 4 - Model Development  
  
Simple Linear regression refers to one independent variable to make a prediction.  
![slr](https://github.com/omkar-334/IBM-DataAnalysis/assets/40126336/170ffd2d-bc57-4124-94d4-6234003811a4)  
Multiple linear regression refers to multiple independent variables to make a prediction.  
![mlr](https://github.com/omkar-334/IBM-DataAnalysis/assets/40126336/117c9c62-4eb8-40d2-bada-93993e2c523d)  
MLR is used to explain the relationship between one continuous target variable Y and two or more predictor X values  
![mvs](https://github.com/omkar-334/IBM-DataAnalysis/assets/40126336/f2453d91-6852-4459-8c34-367f7fe6282b)  
![poly](https://github.com/omkar-334/IBM-DataAnalysis/assets/40126336/d136ab5f-926b-4dd6-9a8c-7acc53bf8c17)  
intercept b0 - lm.intercept_  
slope b1 - lm.coef_  
  
#### Self Evaluative Measures  
  
###### MSE - Mean Squared Error  
  
As MSE increases, the points get further away from the regression line  
  
###### R^2 - Coefficient of Determination (0-1)  
  
The percentage of variation of the target variable that is explained by the linear variable  
How close the data is to the fitted regression line(-ve R^2 due to overfitting)  
=1 good fit  
=0.9 - 90 % of the observed variations can be explained by the independent variables  
  
![rsq](https://github.com/omkar-334/IBM-DataAnalysis/assets/40126336/c5bca474-5909-450e-917b-7e7f06d3c2a0)  
![rsqgraph](https://github.com/omkar-334/IBM-DataAnalysis/assets/40126336/f61745c5-2073-453f-b0d4-081f47f99a33)  
![rsqform](https://github.com/omkar-334/IBM-DataAnalysis/assets/40126336/c5bca474-5909-450e-917b-7e7f06d3c2a0)  
  
### Week 5 - Model Evaluation and Refinement  
  
###### Generalization error  
  
Measure of how well our data does at predicting previously unseen data  
Error we obtain using testing data is approx equal to this error  
  
###### Cross validation  
  
Divide the data into K folds  
  
from sklearn.model_selection import cross_val_score  
scores=cross_val_score(lr, x_data, y_data, cv=5)  
LR - linear regression model  
cv - number of folds  
np.mean(scores)  
  
from sklearn.model_selection import cross_val_predict  
yhat=cross_val_predict(lr2e, x_data, y_data, cv=5)  
Same parameters as score function  
  
###### Model selection  
  
Determine the order of the polynomial to provide the best estimate of the function y(x)  
Overfitting - model is too flexible, fits the noise rather the function.  
Order of the polynomial decreases as the MSE of model decreases.  
Irreductible error.  
Accuracy max when R^2 is closer to 1  
  
###### Ridge regression  
  
A regression that is employed in a Multiple regression model when Multicollinearity occurs.  
Multicollinearity is when there is a strong relationship among the independent variables.  
Ridge regression is very common with polynomial regression.  
  
As alpha increases, coefficients decrease.  
Use cross validation to select alpha  
High alpha - underfitting  
  
###### Validation Data  
  
Train -> Predict -> Calculate R^2  
  
###### Grid Search  
  
Alpha is a hyperparameter  
Split data into 3 parts - training, validation, data  
  
from sklearn.linear_model import Ridge  
from sklearn.model_selection import GridSearchCV  
parameters=[{'alpha':[0.001,0.01,0.1,1,10,100,1000],'normalize':[True,False]}]  
RR=Ridge()  
Grid1=GridSearchCV(RR,paramerters,cv=4)  
Grid1.fit(c_data[['horspower','curb-weight','engine-size','highway-mpg']],y_data)  
Grid1.best_estimator_  
scores=Grid1.cv_results_  
scores['mean_test_score']  
  
for param,mean_val,mean_test in zip(scores['params'],scores['mean_test_score'],scores['mean_train_score']):  
	print(param, "R^2 on test data:", mean_val, "R^2 on train data:", mean_test)  
	
#### Certificate
![Cert](https://user-images.githubusercontent.com/40126336/242241483-816e80b0-cf85-4234-9a6a-3e4c4d98487d.png)
