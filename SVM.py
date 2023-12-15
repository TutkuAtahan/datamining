import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, make_scorer
from sklearn.preprocessing import MinMaxScaler
import time

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

# SVM modeli oluşturma
svm_model = SVC(kernel='linear', random_state=42)

# KFold cross-validation için ayarlar
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# F1 score'u metrik olarak kullanarak cross-validation yapma
f1_scorer = make_scorer(f1_score, average='weighted')

# Cross-validation ile model performansını değerlendirme
start_time = time.time()
cv_accuracy = cross_val_score(svm_model, X, y, cv=kf, scoring='accuracy')
cv_f1 = cross_val_score(svm_model, X, y, cv=kf, scoring=f1_scorer)
end_time = time.time()
cv_time = end_time - start_time

# Sonuçları gösterme
print("Ortalama Accuracy:", cv_accuracy.mean())
print("Ortalama F-measure:", cv_f1.mean())
print("Cross-validation Süresi:", cv_time)
