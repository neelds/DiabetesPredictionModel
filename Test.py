
#Importing the basic librarires for analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import r2_score

#Import the dataset
df =pd.read_csv("C:\\Users\\neeld\Desktop\Study\Tutorial_1\pythonProjectTestDataScientist\diabetes - DS.csv")

#print first five rows from the data set
print(df.head())

#print last five rows from the data set
print(df.head())

#Check the data types of columns
print(df.info())
#Check the null values of columns
print(df.isnull().sum())
#Check the na values of columns
print(df.isna().sum())
#Describe the data set
print(df.describe())

#we can see.. there are no missing values
#two types of data (int,float)

#..........perform the EDA.........
Columns_all = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age','Outcome']

#using histogram identifying the distribution of each columns data
fig = plt.figure(figsize=(3,3))
a = 3  # number of rows
b = 3  # number of columns
c = 1 # initialize plot counter
for i in Columns_all:
    plt.subplot(a, b, c)
    plt.title('{}, subplot: {},{},{}'.format(i, a, b, c))
    plt.xlabel(i)
    plt.hist(x=df[i],bins=45,color='red')
    c = c + 1

plt.tight_layout()
plt.show()
#print the outcome count
print(df.Outcome.value_counts())

#plot that outcome count
count_outcome = pd.value_counts(df['Outcome'],sort=True)
count_outcome.plot(kind='bar',rot=0,)
plt.title('Outcome Distribution')
plt.ylabel('Count')
plt.xlabel('outcome')
plt.show()

#Add the column names into a list
Columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

#Count the Zeros in each rows
for i in Columns:
    count = (df[i] == 0).sum()
    print('Zero count in column :',i,' are :::', count, )

#Zero count in column : Pregnancies  are :::             111
#Zero count in column : Glucose  are :::....               5
#Zero count in column : BloodPressure  are :::            35
#Zero count in column : SkinThickness  are :::           227
#Zero count in column : Insulin  are :::                 374
#Zero count in column : BMI  are :::                      11
#Zero count in column : DiabetesPedigreeFunction  are :::  0
#Zero count in column : Age  are :::                       0

#pregnancies can be have zero values
#But 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin' and 'BMI' can not have zero values..
#Then I will fill them using median value from respective columns median


df['Glucose'] = df['Glucose'].replace(0, df['Glucose'].median())
df['BloodPressure'] = df['BloodPressure'].replace(0, df['BloodPressure'].median())
df['SkinThickness'] = df['SkinThickness'].replace(0, df['SkinThickness'].median())
df['Insulin'] = df['Insulin'].replace(0, df['Insulin'].median())
df['BMI'] = df['BMI'].replace(0, df['BMI'].median())

print('After zero values replace with median---------------------')
for i in Columns:
    count = (df[i] == 0).sum()
    print('Zero count in column :',i,' are :::', count, )

#Now check the outliers
Columns_all = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age','Outcome']

#using boxplot identifying the outliers
fig = plt.figure(figsize=(3,3))
a = 3  # number of rows
b = 3  # number of columns
c = 1 # initialize plot counter
for i in Columns_all:
    plt.subplot(a, b, c)
    plt.title('{}, subplot: {},{},{}'.format(i, a, b, c))
    plt.xlabel(i)
    sns.boxplot(x=df[i])
    c = c + 1

plt.tight_layout()
plt.show()

#we can see 'Glucose' and 'Outcome' columns not included outliers..
#But 'Pregnancies', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction' and 'Age' are included Outliers
#Then we have to remove those outliers from the each columns
#Columns_WithOutlier = ['Pregnancies', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

#now find the outliers in 'Insulin'
q25 = np.percentile(df['Insulin'], 25)
q75 = np.percentile(df['Insulin'], 75)
iqr = q75 - q25
print('Percentiles of: 25th=%.3f, 75th=%.3f, IQR=%.3f' % (q25, q75, iqr))

cut_off = iqr * 1.5
lower = q25 - cut_off
upper = q75 + cut_off

outliers = [x for x in df['Insulin'] if x < lower or x > upper]
print('Identified outliers: %d' % len(outliers))

outliers_remove = [x for x in df['Insulin'] if x >= lower and x <= upper]
print('Non-outlier observations: %d' % len(outliers_remove))

#replace outliers with mean of the 'outliers_remove'
df['Insulin'] = df['Insulin'].replace(outliers,np.mean(outliers_remove))

#now find the outliers in 'Pregnancies'
q25 = np.percentile(df['Pregnancies'], 25)
q75 = np.percentile(df['Pregnancies'], 75)
iqr = q75 - q25
print('Percentiles of: 25th=%.3f, 75th=%.3f, IQR=%.3f' % (q25, q75, iqr))

cut_off = iqr * 1.5
lower = q25 - cut_off
upper = q75 + cut_off

outliers = [x for x in df['Pregnancies'] if x < lower or x > upper]
print('Identified outliers Pregnancies: %d' % len(outliers))

outliers_remove = [x for x in df['Pregnancies'] if x >= lower and x <= upper]
print('Non-outlier observations Pregnancies: %d' % len(outliers_remove))

#replace outliers with mean of the 'outliers_remove'
df['Pregnancies'] = df['Pregnancies'].replace(outliers,np.mean(outliers_remove))

#now find the outliers in 'BloodPressure'
q25 = np.percentile(df['BloodPressure'], 25)
q75 = np.percentile(df['BloodPressure'], 75)
iqr = q75 - q25
print('BloodPressure of: 25th=%.3f, 75th=%.3f, IQR=%.3f' % (q25, q75, iqr))

cut_off = iqr * 1.5
lower = q25 - cut_off
upper = q75 + cut_off

outliers = [x for x in df['BloodPressure'] if x < lower or x > upper]
print('Identified outliers: %d' % len(outliers))

outliers_remove = [x for x in df['BloodPressure'] if x >= lower and x <= upper]
print('Non-outlier observations BloodPressure: %d' % len(outliers_remove))

#replace outliers with mean of the 'outliers_remove'
df['BloodPressure'] = df['BloodPressure'].replace(outliers,np.mean(outliers_remove))

#now find the outliers in 'SkinThickness'
q25 = np.percentile(df['SkinThickness'], 25)
q75 = np.percentile(df['SkinThickness'], 75)
iqr = q75 - q25
print('SkinThickness of: 25th=%.3f, 75th=%.3f, IQR=%.3f' % (q25, q75, iqr))

cut_off = iqr * 1.5
lower = q25 - cut_off
upper = q75 + cut_off

outliers = [x for x in df['SkinThickness'] if x < lower or x > upper]
print('Identified outliers SkinThickness: %d' % len(outliers))

outliers_remove = [x for x in df['SkinThickness'] if x >= lower and x <= upper]
print('Non-outlier observations SkinThickness: %d' % len(outliers_remove))

#replace outliers with mean of the 'outliers_remove'
df['SkinThickness'] = df['SkinThickness'].replace(outliers,np.mean(outliers_remove))

#now find the outliers in 'BMI'
q25 = np.percentile(df['BMI'], 25)
q75 = np.percentile(df['BMI'], 75)
iqr = q75 - q25
print('BMI of: 25th=%.3f, 75th=%.3f, IQR=%.3f' % (q25, q75, iqr))

cut_off = iqr * 1.5
lower = q25 - cut_off
upper = q75 + cut_off

outliers = [x for x in df['BMI'] if x < lower or x > upper]
print('Identified outliers BMI: %d' % len(outliers))

outliers_remove = [x for x in df['BMI'] if x >= lower and x <= upper]
print('Non-outlier observations BMI: %d' % len(outliers_remove))

#replace outliers with mean of the 'outliers_remove'
df['BMI'] = df['BMI'].replace(outliers,np.mean(outliers_remove))

#now find the outliers in 'DiabetesPedigreeFunction'
q25 = np.percentile(df['DiabetesPedigreeFunction'], 25)
q75 = np.percentile(df['DiabetesPedigreeFunction'], 75)
iqr = q75 - q25
print('DiabetesPedigreeFunction of: 25th=%.3f, 75th=%.3f, IQR=%.3f' % (q25, q75, iqr))

cut_off = iqr * 1.5
lower = q25 - cut_off
upper = q75 + cut_off

outliers = [x for x in df['DiabetesPedigreeFunction'] if x < lower or x > upper]
print('Identified outliers: %d' % len(outliers))

outliers_remove = [x for x in df['DiabetesPedigreeFunction'] if x >= lower and x <= upper]
print('Non-outlier observations: %d' % len(outliers_remove))

#replace outliers with mean of the 'outliers_remove'
df['DiabetesPedigreeFunction'] = df['DiabetesPedigreeFunction'].replace(outliers,np.mean(outliers_remove))

#now find the outliers in 'Age'
q25 = np.percentile(df['Age'], 25)
q75 = np.percentile(df['Age'], 75)
iqr = q75 - q25
print('Age of: 25th=%.3f, 75th=%.3f, IQR=%.3f' % (q25, q75, iqr))

cut_off = iqr * 1.5
lower = q25 - cut_off
upper = q75 + cut_off

outliers = [x for x in df['Age'] if x < lower or x > upper]
print('Identified outliers Age: %d' % len(outliers))

outliers_remove = [x for x in df['Age'] if x >= lower and x <= upper]
print('Non-outlier observations Age: %d' % len(outliers_remove))

#replace outliers with mean of the 'outliers_remove'
df['Age'] = df['Age'].replace(outliers,np.mean(outliers_remove))


#After removed outliers drowing the boxplots
fig = plt.figure(figsize=(3,3))
a = 3  # number of rows
b = 3  # number of columns
c = 1 # initialize plot counter
for i in Columns_all:
    plt.subplot(a, b, c)
    plt.title('{}, subplot: {},{},{}'.format(i, a, b, c))
    plt.xlabel(i)
    sns.boxplot(x=df[i])
    c = c + 1

plt.tight_layout()
plt.show()

#Now check the multicoliniarity of the data
# Extreme Multicollinearity occurs whenever the
# independent variables are very high and Correlated
# with one or more other independent variables.

print(df.corr())
sns.heatmap(df.corr(),annot=True,cmap="mako", vmin=0, vmax=1)
plt.title("Correlation Matrix", fontsize=20)
plt.show()

#Check the imbalanced data

count_outcome = pd.value_counts(df['Outcome'],sort=True)

print(count_outcome*100/len(df))
#0    65.104167% =~ 65%
#1    34.895833% =~35%
#Many datasets will have an uneven number of instances in each class,
# but a small difference is usually acceptable. As a rule of thumb,
# if a two-class dataset has a difference of greater than 65% to 35%,
# than it should be looked at as a dataset with class imbalance
# Then we don't need to balance this data set,


# the necessary libraries are imported beginning
#Extracting Independent and dependent Variable

y= df['Outcome']
print(y.head())
x= df.drop(['Outcome'],axis=1)
print(x.head())

# Splitting the dataset into training and test set.
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.2, random_state=0)

# Seeing the split across training and testing datasets
print('Number of records and shape in the original dataset: ', len(x),'--shape :',x.shape)
print('Number of records and shape in the training dataset: ', len(x_train),'--shape :',x_train.shape)
print('Number of records and shape in the testing dataset: ', len(x_test),'--shape :',x_test.shape)
print('Number of records and shape in the original dataset: ', len(y),'--shape :',y.shape)
print('Number of records and shape in the training dataset: ', len(y_train),'--shape :',y_train.shape)
print('Number of records and shape in the testing dataset: ', len(y_test),'--shape :',y_test.shape)

# Now we can scaler standerdization independent variables

st_x= StandardScaler()
x_train= st_x.fit_transform(x_train)
x_test= st_x.transform(x_test)

print(x_test)
print(x_train)

# Now build the model using below algorithms,
# 1) Decision tree
# 2) Random forest
# 3) Support vector machine
# 4) Gradient boosting
# 5) Logistic regression
# 6) K-nearest neighbors

DT = DecisionTreeClassifier()
RF = RandomForestClassifier()
SVM = SVC()
GB = GradientBoostingClassifier()
LR = LogisticRegression()
KNN = KNeighborsClassifier()

#Create list of algorithms
li = [DT,RF,SVM,GB,LR,KNN]
d = {}


for i in li:
    # fitting the training data set into the the each model
    i.fit(x_train,y_train)
    #predict the outcome using testing data set
    ypred = i.predict(x_test)
    #print the r squar values for each algorithms
    print(i,":",r2_score(ypred,y_test)*100)
    d.update({str(i):i.score(x_test,y_test)*100})

# now comparison of the models and find the best model from the list of model

plt.figure(figsize=(20,6))
plt.title("Algorithm vs Accuracy")
plt.xlabel("Algorithm")
plt.ylabel("Accuracy")
plt.plot(d.keys(),d.values(),marker='o',color='black')
plt.show()

# We can see the best model is "Gradient boosting" clasification model.

