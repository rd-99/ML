import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
#import matplotlib.mlab as mlab
import seaborn
import warnings
%matplotlib inline
warnings.filterwarnings("ignore")

data = pd.read_csv('../input/creditcardfraud/creditcard.csv')
df = pd.DataFrame(data)
(df.head())
df.describe()

df.isnull().sum()
#no missing values

df_corr = df.corr()
plt.figure(figsize=(15,10))
seaborn.heatmap(df_corr, cmap="YlGnBu")
plt.title('Heatmap correlation')
plt.show()

print("Non-Frauduant transactions are - ", df["Class"].value_counts()[0]/len(df["Class"]) * 100,"% of dataset")
print("Frauduant transactions are - ", df["Class"].value_counts()[1]/len(df["Class"]) * 100,"% of dataset")
print(df["Class"].value_counts()[1],df["Class"].value_counts()[0])

fraud = df[df["Class"] == 1]
print('Mean for fraud-transactions- ',fraud["Amount"].mean(),", std for fraud-transactions-", fraud["Amount"].std())
non_fraud = df[df["Class"] == 0]
print('Mean for non-fraud-transactions- ',non_fraud["Amount"].mean()," std for non-fraud-transactions-", non_fraud["Amount"].std())
print("Median are " ,fraud["Amount"].median() ," and ", non_fraud["Amount"].median(), " respectively" )
sample_non_fraud = non_fraud.sample(492)
print(fraud["Amount"].max() , non_fraud["Amount"].max())

fraud_df,non_fraud_df = fraud.loc[fraud['Amount'] >= 1000], non_fraud.loc[non_fraud['Amount'] >= 2000] 
print(len(fraud_df.value_counts()) , len(non_fraud_df.value_counts()))

import seaborn as sns
sns.FacetGrid(non_fraud_df, hue="Class", size=5) \
   .map(sns.distplot, "Amount") \
   .add_legend();
sns.FacetGrid(fraud_df, hue="Class", size=5) \
   .map(sns.distplot, "Amount") \
   .add_legend();
plt.show()
sns.boxplot(x='Class',y='Amount', data=df)
plt.show()

!pip install dataprep
from dataprep.eda import plot, plot_correlation, create_report, plot_missing
plot(df)

create_report(df)

from sklearn.preprocessing import RobustScaler
rob_scaler = RobustScaler()
df['scaled_amount'] = rob_scaler.fit_transform(df['Amount'].values.reshape(-1,1))
df['scaled_time'] = rob_scaler.fit_transform(df['Time'].values.reshape(-1,1))
df.drop(['Time','Amount'], axis=1, inplace=True)

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit

print('No Frauds', round(df['Class'].value_counts()[0]/len(df) * 100,2), '% of the dataset')
print('Frauds', round(df['Class'].value_counts()[1]/len(df) * 100,2), '% of the dataset')

X = df.drop('Class', axis=1)
y = df['Class']

sss = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)

for train_index, test_index in sss.split(X, y):
    print("Train:", train_index, "Test:", test_index)
    original_Xtrain, original_Xtest = X.iloc[train_index], X.iloc[test_index]
    original_ytrain, original_ytest = y.iloc[train_index], y.iloc[test_index]
original_Xtrain = original_Xtrain.values
original_Xtest = original_Xtest.values
original_ytrain = original_ytrain.values
original_ytest = original_ytest.values

# This is explicitly used for undersampling.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)
X_train = X_train.values
X_test = X_test.values
y_train = y_train.values
y_test = y_test.values

import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy

n_inputs = X_train.shape[1]

undersample_model = Sequential([
    Dense(n_inputs, input_shape=(n_inputs, ), activation='relu'),
    Dense(32, activation='relu'),
    Dense(2, activation='softmax')
])
undersample_model.summary()
undersample_model.compile(Adam(lr=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
undersample_model.fit(X_train, y_train, validation_split=0.2, batch_size=25, epochs=5, shuffle=True, verbose=2)
undersample_predictions = undersample_model.predict(original_Xtest, batch_size=200, verbose=0)
undersample_fraud_predictions = undersample_model.predict_classes(original_Xtest, batch_size=200, verbose=0)

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
sm = SMOTE( sampling_strategy='auto',random_state=4)
Xsm_train, ysm_train = sm.fit_resample(original_Xtrain, original_ytrain)

n_inputs = Xsm_train.shape[1]

oversample_model = Sequential([
    Dense(n_inputs, input_shape=(n_inputs, ), activation='relu'),
    Dense(32, activation='relu'),
    Dense(2, activation='softmax')
])
oversample_model.compile(Adam(lr=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
oversample_model.fit(Xsm_train, ysm_train, validation_split=0.2, batch_size=300, epochs=5, shuffle=True, verbose=2)
oversample_predictions = oversample_model.predict(original_Xtest, batch_size=200, verbose=0)
oversample_fraud_predictions = oversample_model.predict_classes(original_Xtest, batch_size=200, verbose=0)
