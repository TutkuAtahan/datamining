import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import MinMaxScaler
import time
from sklearn.metrics import make_scorer


# Veri setini yükleme
df = pd.read_csv('veriseti.csv', delimiter=';')

# Eksik değerleri kontrol etme
print(df.isnull().sum())

# Kategorik değişkenleri işleme (örneğin, One-Hot Encoding)
df = pd.get_dummies(df, columns=['Marital status', 'Nacionality'])

# Eksik değerleri doldurma sadece sayısal özellikler için
numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

# Özellikleri ölçeklendirme
scaler = MinMaxScaler()
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

# Bağımsız değişkenler ve hedef değişkeni ayırma
X = df.drop('Target', axis=1)
y = df['Target']

# Gradient Boosting Classifier modeli oluşturma
gb_model = GradientBoostingClassifier(random_state=42)  

# Veri setini eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# KFold cross-validation için ayarlar
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Eğitim süresini ölçme (KFold ile)
start_train_time = time.time()
for train_index, _ in kf.split(X):
    X_train_kf, y_train_kf = X.iloc[train_index], y.iloc[train_index]
    gb_model.fit(X_train_kf, y_train_kf)
end_train_time = time.time()
training_time_kf = end_train_time - start_train_time

# Test süresini ölçme (KFold ile)
start_test_time = time.time()
for _, test_index in kf.split(X):
    X_test_kf, y_test_kf = X.iloc[test_index], y.iloc[test_index]
    gb_model.predict(X_test_kf)
end_test_time = time.time()
testing_time_kf = end_test_time - start_test_time

# F1 score'u metrik olarak kullanarak cross-validation yapma
f1_scorer = make_scorer(f1_score, average='weighted')
start_cv_time = time.time()
cv_f1 = cross_val_score(gb_model, X, y, cv=kf, scoring=f1_scorer)
end_cv_time = time.time()
cv_time = end_cv_time - start_cv_time

# Sonuçları gösterme
print("Ortalama F-measure:", cv_f1.mean())
print("Eğitim Süresi (KFold ile):", training_time_kf)
print("Test Süresi (KFold ile):", testing_time_kf)
print("Cross-validation Süresi:", cv_time)
