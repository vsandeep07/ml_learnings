import pandas as pd
import numpy as np
from sklearn import pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


def load_data(path):
    df = pd.read_csv(path)
    return df

df = load_data(r"C:\Users\YERSANSAI\OneDrive\Desktop\practise\TD.csv")

# inspecting data 

def inspect_data(df):
    print("Dataframe Head:")
    df.head()
    print("Dataframe Info:")
    df.info()
    print("Dataframe Description:")
    df.describe()
    print("Missing Values:")
    df.isnull().sum()


inspect_data(df)

# Seperating the data into features

def feature_seperation(df,target_column):
    x = df.drop([target_column],axis  = 1)
    y = df[target_column]

    return x,y


x,y = feature_seperation(df,'Survived')

# Removing the non useful features from data 

def remove_unnecessary_col(x):
    unnecessary_cols = ['PassengerId', 'Name', 'Ticket', 'Cabin']
    x = x.drop(unnecessary_cols, axis=1)
    return x

x = remove_unnecessary_col(x)

# identifying the categorical and numerical columns
def identify_col(x):
    num_col = x.select_dtypes(include=['float64', 'int64']).columns
    cat_col = x.select_dtypes(include=['object', 'string']).columns
    return num_col, cat_col

numerical_columns, categorical_columns = identify_col(x)

# create preprocessing function 

def create_preprocess(numerical_columns,catgeorical_columns):

    num_col = Pipeline(steps = [
        ('impute',SimpleImputer()),
        ('scaler',StandardScaler())
    ])

    cat_col = Pipeline(steps = [
        ('impute',SimpleImputer(strategy = 'most_frequent')),
        ('OHE',OneHotEncoder(sparse_output = False,handle_unknown = 'ignore'))
    ])

    preprocessor = ColumnTransformer([
        ("numerical", num_col, numerical_columns),
        ("categorical", cat_col, categorical_columns)
    ])

    return preprocessor


preprocessor =  create_preprocess(numerical_columns,categorical_columns)


# splitting data
def split_data(x, y, test_size=0.3, random_state=42):
    return train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state
    )

x_train, x_test, y_train, y_test = split_data(x, y)

# creating model pipeline fucntion 

def pipeline_model(preprocessor,model):
    pipeline = Pipeline(steps = [
        ('preprocessor',preprocessor),
        ('model',model)
    ])

    return pipeline

pipeline_rf = pipeline_model(preprocessor,RandomForestClassifier(n_estimators = 100,random_state = 42))
pipeline_ad = pipeline_model(preprocessor,AdaBoostClassifier(n_estimators = 100,random_state = 42))

# trianing the model

def train_model(pipeline,x_train,y_train):
    pipeline.fit(x_train,y_train)
    return pipeline

rf_model = train_model(pipeline_rf,x_train,y_train)
ad_model = train_model(pipeline_ad,x_train,y_train)

# evaluating the model 

def model_evaluation(model,x_test,y_test):
    y_pred = model.predict(x_test)

    metrics  = {
        'accuracy': accuracy_score(y_test,y_pred),
        'precision': precision_score(y_test,y_pred),
        'recall': recall_score(y_test,y_pred),
    }
    return metrics

rf_model_metrics = model_evaluation(rf_model,x_test,y_test)
ad_model_metrics = model_evaluation(ad_model,x_test,y_test)

print("??????????????????????????????????")

print("Random Forest Metrics:")
print(rf_model_metrics)

print("??????????????????????????????????")
print("\nAdaBoost Metrics:")
print(ad_model_metrics)