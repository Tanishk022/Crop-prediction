# %%
import pandas as pd
import numpy as np

# %%
df=pd.read_csv("clean_crop_data.csv")
df

# %%
df.shape

# %%
df.isnull().sum()

# %%
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['label'] = le.fit_transform(df['label'])


# %%
df.head()

# %%
df['label'].unique()

# %% [markdown]
# ### Model Train Test Split

# %%
# Input features
x = df.drop('label', axis=1)
# Target variable
y = df['label']  

# %%
x

# %%
y

# %%
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=33)

# %%
x_train

# %%
x_test

# %%
y_test

# %%
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)


# %%
x_train_scaled

# %%
x_test_scaled

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import xgboost as xb
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# %%
# Logistic Regression Model
lrc = LogisticRegression(max_iter=300)
lrc

# %%
lrc.fit(x_train_scaled, y_train)

# %%
y_pred =lrc.predict(x_test)
y_pred

# %% [markdown]
# ### Confusion Metric,Accuracy,Classification Report

# %%
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report

# %%
accuracy_lrc = accuracy_score(y_test, y_pred)
print("Logistic Regression Accuracy:", accuracy_lrc)

# %%
confusion_matrix_LR=confusion_matrix(y_test,y_pred)
print("Confusion Matrix - Logistic Regression",confusion_matrix_LR)

# %%
print(classification_report(y_test,y_pred))

# %% [markdown]
# ### SVM Model

# %%
# SVM Model
svm_model = SVC()
svm_model.fit(x_train_scaled, y_train)


# %%
y_pred_svm = svm_model.predict(x_test_scaled)

# %%
y_pred_svm

# %%
print(classification_report(y_test, y_pred_svm))
print("SVM modelAccuracy :", accuracy_score(y_test, y_pred_svm))
cm_SVM = confusion_matrix(y_test, y_pred_svm)
print(cm_SVM)

# %% [markdown]
# #### Decision Tree

# %%
dt_model = DecisionTreeClassifier(criterion="entropy")
dt_model.fit(x_train, y_train)

# %%
y_pred_dt = dt_model.predict(x_test)
y_pred_dt

# %%
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print("Decision Tree Accuracy:", accuracy_dt)


# %%
cm_DT = confusion_matrix(y_test, y_pred_dt)
print("Decision Tree Confusion matric:", cm_DT)


# %%
print("Classification Report for Decision Tree:\n")
print(classification_report(y_test, y_pred_dt))

# %% [markdown]
# ### Random Forest

# %%
rf_model = RandomForestClassifier()
rf_model.fit(x_train, y_train)

# %%
y_pred_rf = rf_model.predict(x_test)
y_pred_rf

# %%
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print("Random Forest Accuracy:", accuracy_rf)
cm_RF = confusion_matrix(y_test, y_pred_rf)
print("Random Forest confusion matrix",cm_RF)

print("Classification Report for Random Forest:\n")
print(classification_report(y_test, y_pred_rf))

# %% [markdown]
# #### XGBoost

# %%
import xgboost as xb

# %%
xgb_model = xb.XGBClassifier()
xgb_model.fit(x_train, y_train)
# classifier3=xb.XGBClassifier()
# classifier3.fit(X_train,y_train)

# %%
y_pred_xb=xgb_model.predict(x_test)
print("confusion of XGBOOST",confusion_matrix(y_test,y_pred_xb))
print("accuray of XGBOOST",accuracy_score(y_test,y_pred_xb))
print("report of XGBOOST",classification_report(y_test,y_pred_xb))

# %% [markdown]
# ### All model accuracy comperision

# %%
# y_pred
# y_pred_svm
# y_pred_dt
# y_pred_rf
# y_pred_xb

model_accuracy = {
    "Logistic Regression": accuracy_score(y_test, y_pred),
    "SVM": accuracy_score(y_test, y_pred_svm),
    "Decision Tree": accuracy_score(y_test, y_pred_dt),
    "Random Forest": accuracy_score(y_test, y_pred_rf),
    "XGBoost": accuracy_score(y_test, y_pred_xb)
}

accuracy_df = pd.DataFrame(list(model_accuracy.items()), columns=["Model", "Accuracy"])
print(accuracy_df)

# %%
LR = accuracy_score(y_test, y_pred)
SVM=accuracy_score(y_test, y_pred_svm)
DT= accuracy_score(y_test, y_pred_dt)
RF= accuracy_score(y_test, y_pred_rf)
Xb=accuracy_score(y_test, y_pred_xb)

# %%
lst = ['LR','SVM','DT','RF','Xb']
best_model = lst[0]
for i in range(len(lst)-1):
    if lst[i] <= lst[i+1]:
        best_model = lst[i+1]
print(best_model)

# %%
print(best_model)

# %%
import pickle

file = open('crop_model_1.pkl', 'wb')
# dump information to that file
pickle.dump(best_model, file)

# %%
file
