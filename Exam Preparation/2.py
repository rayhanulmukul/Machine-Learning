import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('BrestCancer.csv')

df = df.fillna(df.mean(numeric_only=True))

lr = LabelEncoder()
df['target'] = lr.fit_transform(df['diagnosis'])
target_names = list(lr.classes_)

X = df.drop(columns=('id', 'dianosis', 'target'))
y = df['target']
feature_names = X.columns

sns.countplot(df, x='diagnosis', palette='coolwarm')
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)


rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))

print("SVM Confusion Matrix:\n", confusion_matrix(y_test, y_pred_svm))
print("Random Forest Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))

print("\nSVM Classification Report:\n", classification_report(y_test, y_pred_svm, target_names=target_names))
print("Random Forest Classification Report:\n", classification_report(y_test, y_pred_rf, target_names=target_names))


importance = np.abs(rf_model.feature_importances_)
indics = np.argsort(importance)[::-1]

sns.barplot(importance[indics], feature_names[indics], palette='viridis')
plt.show()

importance = np.abs(svm_model.coef_).flatten()
indics = np.argsort(importance)[::-1]

sns.barplot(importance[indics], feature_names[indics], palette='viridis')
plt.show()