# customer churn model

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

'''
# get Random forest model (n_estimators is number of descision trees used)
model = RandomForestClassifier(n_estimators=100, random_state=0)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

# evaluate and test accuracy of model
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

sns.heatmap(
    conf_matrix,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("results/confusion_matrix_randomForest.png")
plt.close()

print(class_report)
print(accuracy)

# visualize the importances of each feature
importances = model.feature_importances_
indices = np.argsort(importances)[::-1] # reverses order so most to least important
names = [features[i] for i in indices]

plt.figure(figsize=(10,6))
plt.title("Feature importance")
plt.barh(range(x.shape[1]), importances[indices])
plt.yticks(range(x.shape[1]), names)
plt.savefig("results/feature_importances_randomForest.png")
plt.close()

'''



# loads dataframe and returns
def loadData():
    df = pd.read_csv("data/Churn_Modelling.csv")
    return df


# Gives the basic information about the dataset
def understandDataset(df):
    print("\nDataset shape:")
    print(df.shape)

    print("\nFirst five rows:")
    print(df.head())

    print("\nColumn information:")
    print(df.info())

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nTarget distribution:")
    print(df["Exited"].value_counts())

    print("\nTarget proportions:")
    print(df["Exited"].value_counts(normalize=True))


# creates a barchart to show the ratio of customers churning to cutomers staying
def plotChurnDist(df):
    plt.figure(figsize=(7,5))
    sns.countplot(data=df, x="Exited")
    plt.title("Customer Churn Distribution")
    plt.xlabel("Exited")
    plt.ylabel("Number of customers")
    plt.tight_layout()
    plt.savefig("results/churn_distribution.png")
    plt.close()


# creates a barplot to represent the proportion number of customers churned per categogy of a specific feature 
def plotChurnByCategory(df, col):
    churnRate = df.groupby(col)['Exited'].mean()
    plt.figure(figsize=(7,5))
    sns.barplot(x=churnRate.index, y=churnRate.values)
    plt.title(f"Churn Rate by {col}")
    plt.xlabel(col)
    plt.ylabel('Churn rate')
    plt.tight_layout()
    plt.savefig(f"results/churn_by_{col.lower()}.png")
    plt.close()


# creates a 
def plotChurnForNumerical(df, col):
    plt.figure(figsize=(7,5))
    sns.boxplot(data=df, x='Exited', y=col)
    plt.title(f"{col} distribution by churn")
    plt.xlabel("Exited")
    plt.ylabel(col)
    plt.tight_layout()
    plt.savefig(f"results/churn_by_{col.lower()}.png")
    plt.close()


# Plots graphs to explore the proportion of customers churning as well as how each attribute affects the churn rate
def conductExploratoryAnalysis(df):
    plotChurnDist(df)
    categorical = ['Gender', 'HasCrCard', 'IsActiveMember', 'Geography']
    numerical = ['CreditScore',  'Age', 'Tenure', 'Balance', 'NumOfProducts']

    for col in categorical:
        plotChurnByCategory(df, col)

    for col in numerical:
        plotChurnForNumerical(df, col)


# only trains on relevant features and splits the target exited column
def splitFeaturesFromTarget(df):
    features = ['CreditScore', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'Geography']
    x = df[features]
    y = df["Exited"]
    return x,y


# splits the data into test and training data and uses stratify to maintain approximately same proportion of churned to non-churned
def splitTestTrain(x,y):
    xTrain, xTest, yTrain, yTest = train_test_split(x,y,test_size=0.2, stratify=y)
    return xTrain, xTest, yTrain, yTest


# makes a transformer to scale numerical features and to encode categorical values
def makePreprocessor(rf=False):
    numericalFeatures = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember']
    categoricalFeatures = ['Gender', 'Geography']

    if rf:
        numericalTransformer = ("numerical", "passthrough", numericalFeatures)
    else:
        numericalTransformer = ("numerical", StandardScaler(), numericalFeatures)
    categoricalTransformer = ("categorical", OneHotEncoder(drop='first', handle_unknown='ignore'), categoricalFeatures)

    preprocessor = ColumnTransformer(transformers=[numericalTransformer, categoricalTransformer])
    return preprocessor


# makes model pipeline for logistic regression to prevent data leakage, and streamline workflow
def createLogisticRegressionPipeline():
    preprocessor = makePreprocessor()
    classifier = LogisticRegression(random_state=42,max_iter=1000)
    model = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', classifier)])
    return model


# creates a model pipeline for random forest
def createRandomForestPipeline():
    preprocessor = makePreprocessor(rf=True)
    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    model = Pipeline(steps=[('preprocessor',preprocessor), ('classifier', classifier)])
    return model 

# gets the predicted values and probabilities from the model
def makePredictions(model, x_test):
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]
    return predictions, probabilities


def evaluateModel(y_test, preds, probs):
    accuracy = accuracy_score(y_true=y_test, y_pred=preds)
    precision = precision_score(y_true=y_test, y_pred=preds)
    recall = recall_score(y_true=y_test, y_pred=preds)
    f1 = f1_score(y_true=y_test, y_pred=preds)
    roc_auc = roc_auc_score(y_true=y_test, y_score=probs)

    print(f"Accuracy:  {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1-score:  {f1:.3f}")
    print(f"ROC-AUC:   {roc_auc:.3f}")

    classificationReport = classification_report(y_true=y_test, y_pred=preds)

    print("\nClassification report:")
    print(classificationReport)

    return {"Accuracy" : accuracy,
            "Precision" : precision,
            "Recall" : recall,
            "F1 score" : f1,
            "ROC-AUC" : roc_auc}


# presents results of each model in a new dataframe with metrics rounded to 3dp
def compareModels(logistic_results, rf_results):
    results = pd.DataFrame({"Logistic Regression" : logistic_results, "Random Forest" : rf_results})
    print("\nModel Comparison:")
    print(results.round(3))
    return results


def createVisualisations():
    ...

def main():
    df = loadData()

    # does inital analysis of data
    understandDataset(df)
    conductExploratoryAnalysis(df)

    # splits the relevant features of the database into test and train sets
    x,y = splitFeaturesFromTarget(df)
    x_train, x_test, y_train, y_test = splitTestTrain(x,y)

    # logistic regression model
    logisticModel = createLogisticRegressionPipeline()
    logisticModel.fit(x_train, y_train)
    predictions_logictic, probabilities_logictic = makePredictions(logisticModel, x_test)

    print("\nLogistic regression results:")
    logisticResults = evaluateModel(y_test, predictions_logictic, probabilities_logictic)

    # random forest model
    randomForestModel = createRandomForestPipeline()
    randomForestModel.fit(x_train, y_train)
    predictions_randomForest, probabilities_randomForest = makePredictions(randomForestModel, x_test)

    print("\nRandom forest results:")
    rfResults = evaluateModel(y_test, predictions_randomForest, probabilities_randomForest)

    # comparison of the models for model selection
    results = compareModels(logisticResults, rfResults)



if __name__ == "__main__":
    main()