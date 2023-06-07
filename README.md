# Fraudulent-Transaction-Monitoring-System

Data Cleaning:
The code starts by importing the necessary libraries: pandas, numpy, matplotlib, seaborn, train_test_split, and LogisticRegression.
The dataset is loaded using pd.read_csv from the 'Fraud.csv' file.
The first 10 rows of the dataset are displayed using data.head(10).
Unnecessary columns ('type', 'nameOrig', 'nameDest') are dropped using data.drop(columns=columns_to_remove).
Duplicated rows are removed using data.drop_duplicates().
The count plot of the 'isFraud' column is displayed using sns.countplot(data['isFraud']).
Information about the dataset is displayed using data.info().
Missing values in the dataset are checked using data.isnull().values.any().
The last 5 rows of the dataset are displayed using data.tail().
The count of 'isFraud' values (0 and 1) is displayed using data['isFraud'].value_counts().
Separate datasets are created for legitimate (0) and fraudulent (1) transactions using boolean indexing: legit = data[data.isFraud == 0] and fraud = data[data.isFraud == 1].
Descriptive statistics of the 'amount' column are displayed for both legitimate and fraudulent transactions using legit.amount.describe() and fraud.amount.describe().
Mean values for each column are calculated based on 'isFraud' using data.groupby('isFraud').mean().

Correlation Analysis and Variable Selection:
The correlation matrix of the dataset is computed using data.corr().abs().
A correlation threshold of 0.8 is set.
Highly correlated variables are identified by comparing the upper triangle of the correlation matrix with the threshold.
One variable from each correlated pair is selected to remove redundancy.
The dataset is updated with the selected variables.

Fraud Detection Model:
The target variable ('isFraud') and features are selected.
The features are stored in the variable 'X', and the target variable is stored in the variable 'y'.
The 'X' and 'y' variables are printed to display the selected data.
