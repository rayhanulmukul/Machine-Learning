import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv('iris.csv')
display(df.head())

feature_colums = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
X = df[feature_colums].values
y = df['species'].values

sns.pairplot(df, x_vars=feature_colums, hue='species', diag_kind='kde', markers=['o', 's', 'D'], palette='Set1')
plt.show()

X_scaled = MinMaxScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
y_pred_log = lr_model.predict(X_test)

nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
y_pred_nb = nb_model.predict(X_test)

print('\nLogistic Regression Accuracy', accuracy_score(y_test, y_pred_log))
print('\nMultinomialNaiveBias Accuracy', accuracy_score(y_test, y_pred_nb))

print('\nLogistic Regression confusion_matrix\n', confusion_matrix(y_test, y_pred_log))
print('\nMultinomialNaiveBias confusion_matrix\n', confusion_matrix(y_test, y_pred_nb))

print('\nLogistic Regression classification_report\n', classification_report(y_test, y_pred_log))
print('\nMultinomialNaiveBias classification_report\n', classification_report(y_test, y_pred_nb))

importance = np.mean(np.abs(lr_model.coef_), axis=0)
imp_df = pd.DataFrame({'Feature': feature_colums, 'Importances': importance}).sort_values('Importances')

plt.barh(imp_df['Feature'], imp_df['Importances'])
plt.xlabel('Mean | Coefficient')
plt.ylabel('Importance')
plt.show()


log_prob = nb_model.feature_log_prob_
importance = np.abs(log_prob[0] - log_prob[1])
imp_df = pd.DataFrame({'Feature': feature_colums, 'Importances': importance}).sort_values('Importances')

plt.barh(imp_df['Feature'], imp_df['Importances'])
plt.xlabel('Mean | Coefficient')
plt.ylabel('Importance')
plt.show()
