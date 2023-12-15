import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Veri setini yükleme
df = pd.read_csv('veriseti.csv', delimiter=';')  # Veri setinin dosya yolunu düzgün bir şekilde belirtin

# Veri setinin genel bilgilerini gözden geçirme
print(df.info())

# Bağımsız değişkenler ve hedef değişkeni ayırma
X = df.drop('Target', axis=1)
y = df['Target']

# Veri setini eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Naive Bayes modeli oluşturma
nb_model = GaussianNB()

# Modeli eğitim verileri ile eğitme
nb_model.fit(X_train, y_train)

# Test seti üzerinde tahmin yapma
predictions = nb_model.predict(X_test)

# Modelin performansını değerlendirme
accuracy = accuracy_score(y_test, predictions)
print("Naive Bayes ile elde edilen doğruluk:", accuracy)
