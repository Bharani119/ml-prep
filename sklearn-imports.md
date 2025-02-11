### most used scikit-learn import stmts

When using scikit-learn, there are several commonly used import statements that you might find in many machine learning projects. Here are some of the most frequently used imports:

1. **General Imports**:
   ```python
   import numpy as np
   import pandas as pd
   ```

2. **Model Imports**:
   - For classification:
     ```python
     from sklearn.ensemble import RandomForestClassifier
     from sklearn.linear_model import LogisticRegression
     from sklearn.svm import SVC
     from sklearn.tree import DecisionTreeClassifier
     from sklearn.neighbors import KNeighborsClassifier
     ```

   - For regression:
     ```python
     from sklearn.ensemble import RandomForestRegressor
     from sklearn.linear_model import LinearRegression
     from sklearn.svm import SVR
     from sklearn.tree import DecisionTreeRegressor
     ```

3. **Preprocessing**:
   ```python
   from sklearn.model_selection import train_test_split
   from sklearn.preprocessing import StandardScaler
   from sklearn.preprocessing import OneHotEncoder
   from sklearn.impute import SimpleImputer
   ```

4. **Model Evaluation**:
   ```python
   from sklearn.metrics import accuracy_score
   from sklearn.metrics import confusion_matrix
   from sklearn.metrics import classification_report
   from sklearn.metrics import mean_squared_error
   from sklearn.metrics import r2_score
   ```

5. **Pipeline and Feature Selection**:
   ```python
   from sklearn.pipeline import Pipeline
   from sklearn.feature_selection import SelectKBest
   from sklearn.feature_selection import f_classif
   ```

6. **Cross-Validation**:
   ```python
   from sklearn.model_selection import cross_val_score
   from sklearn.model_selection import KFold
   ```

These imports cover a wide range of functionalities in scikit-learn, from model training and evaluation to data preprocessing and feature selection. Depending on your specific use case, you may need to import additional modules or classes.