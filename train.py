import pandas as pd
import joblib

from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report

data = pd.read_csv('winequality-red.csv')
data.isnull().sum()

imputer = KNNImputer(n_neighbors=5)
data[data.columns] = imputer.fit_transform(data)
data['quality'] = (data['quality'] >= 7).astype(int) # convert quality to 1's or 0's

x = data.drop(columns=['quality'])
y = data['quality']

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)

smote = SMOTE(random_state=42)
x_train_bal, y_train_bal = smote.fit_resample(x_train, y_train)

model = RandomForestClassifier(n_estimators=200, max_depth=10, class_weight='balanced', random_state=42)
model.fit(x_train_bal, y_train_bal)

y_pred = model.predict(x_test)
y_prob = model.predict_proba(x_test)[:, 1]
prediction = model.predict(x_test[0].reshape(1, -1))[0]
confidence = model.predict_proba(x_test[0].reshape(1, -1))[0][prediction]

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')

