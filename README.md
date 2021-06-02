# Credit-Card-Issue-Prediction
logistic regression model to predict whom to be issued a credit card


# Sample Data
ID	      Gender	Age	Region_Code	Occupation	Channel_Code	Vintage	Credit_Product	Avg_Account_Balance	Is_Active	Is_Lead
NNVBBKZB	Female	73	RG268	      Other	        X3	          43	      No	              1045696	          No	      0
IDD62UNG	Female	30	RG277	      Salaried	    X1	          32	      No	              581988	          No	      0
HD3DSEMC	Female	56	RG268	      Self_Employed	X3	          26	      No	              1484315	          Yes	      0
BF3NC7KV	Male	  34	RG270	      Salaried	    X1	          19	      No	              470454	          No	      0
ETQCZFEJ	Male	  62	RG282	      Other	        X3	          20		                      1056750	          Yes	      1

There are NaN Values and categorical values in the data


# Handling NaN
We used Random Sample Imputation to replace the NaN with the non NaN values in the data because the data missing is random.


# Handling Categorical Variables

we used Frequency Encoding for 	Region_Code, Occupation, Channel_Code to replace the categorical data with their count as there are too many categories for which if we use OneHotEncoding it causes curse of dimentionality.
def freq_enco(d,variable):
    c = d[variable].value_counts().to_dict()
    d[variable] = d[variable].map(c)


we use OneHotEncoding for Credit_Product, Is_Active as there are only two categories of data.


# Spliltting the data
The data is split into 70% for training and 30% for testing as X_train,x_test,y_train, y_test


# Handling Skewness
def plot_data(df,feature):
    plt.figure(figsize=(10,6))
    plt.subplot(1,2,1)
    df[feature].hist()
    plt.subplot(1,2,2)
    stat.probplot(df[feature],dist = "norm", plot = pylab)
    plt.show()
The variables Age, Vintage, Avg_Account_Balance are Right Skewed so we are using Logarithmic Transformation to change it to Normal Distribution
d["Avg_Account_Balance"] = np.log(d["Avg_Account_Balance"])

# Handling Bias
The Output is biased with 0: 131198, 1: 40809
so we use Over Sampling to reduce bias in the data and also to prevent data loss caused by Under Sampling

os = RandomOverSampler(0.5)
x_train_ns, y_train_ns = os.fit_resample(x_train,y_train)

# Creating and Training model
Here we chose Logistic Regression as this is a Binary Classification Data.

from sklearn.linear_model import LogisticRegression
regressor = LogisticRegression()
regressor.fit(x_train,y_train)

# Predicting the Output from Test data and comparing it with original to calculate the accuracy
y_pred = regressor.predict(x_test)
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)
