import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import MinMaxScaler
import time

# Veri setini yükleme
df = pd.read_csv('veriseti.csv', delimiter=';')  # Veri setinin dosya yolunu düzgün bir şekilde belirtin
# Veri setinin başını inceleme
print(df.head())
# Veri setinin genel bilgilerini gözden geçirme
print(df.info())

# Eksik değerleri kontrol etme
print(df.isnull().sum())

# Kategorik değişkenleri işleme (örneğin, One-Hot Encoding)
df = pd.get_dummies(df, columns=['Marital status', 'Nacionality'])

# Eksik değerleri doldurma sadece sayısal özellikler için
numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

# Özellikleri ölçeklendirme (örneğin, Min-Max ölçeklendirme veya Standart ölçeklendirme)
scaler = MinMaxScaler()
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

# Eksik değerleri tekrar kontrol etme
print(df.isnull().sum())

# Bağımsız değişkenler ve hedef değişkeni ayırma
X = df.drop('Target', axis=1)
y = df['Target']

# Veri setini eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Gradient Boosting Classifier modeli oluşturma
gb_model = GradientBoostingClassifier(random_state=42)  # Gradient Boosting Classifier modelini oluşturuyoruz

# Modeli eğitim verileri ile eğitme
gb_model.fit(X_train, y_train)

# Test seti üzerinde tahmin yapma
predictions = gb_model.predict(X_test)

# Eğitim süresini ölçme
start_time = time.time()
gb_model.fit(X_train, y_train)
end_time = time.time()
training_time = end_time - start_time

# Tahmin süresini ölçme
start_time = time.time()
y_pred = gb_model.predict(X_test)
end_time = time.time()
prediction_time = end_time - start_time

# Accuracy ve F-measure değerlerini hesaplama
accuracy = accuracy_score(y_test, y_pred)
f_measure = f1_score(y_test, y_pred, average='weighted')

print("Model Doğruluğu:", accuracy)
print("F-measure değeri:", f_measure)
print("Eğitim süresi:", training_time)
print("Tahmin süresi:", prediction_time)
