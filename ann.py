# Veri setinin hazırlanması ve preprocessing
from keras.datasets import mnist # mnist'i yükle
from keras.utils import to_categorical # kategorik verilere çevir
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential # sıralı model
from keras.layers import Dense # bağlı katmanlar

from keras.models import load_model # modeli yükle

import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

#mist veri setini yükle ,eğitim ve test veri seti olarak ayrı ayrı yükle
(x_train,y_train),(x_test,y_test)=mnist.load_data()
print("Eğitim verisi boyutu:", x_train.shape)
print("Test verisi boyutu:", x_test.shape)
print("Eğitim etiketleri boyutu:", y_train.shape)
print("Test etiketleri boyutu:", y_test.shape)

# İlk eğitim görüntüsünün piksel değerlerini yazdırma
print("İlk eğitim görüntüsü:\n", x_train[0])
print("İlk eğitim etiketi:", y_train[0])
#ilk birk kaç örneği görselleştir
plt.figure(figsize=(10,5))

for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(x_train[i],cmap="gray")
    plt.title(f"label:{y_train[i]}")
    plt.axis("off")
plt.show()    

# Veri setini normalize edelim, 0-255 aralığındaki piksel değerlerini 0-1 arasına ölçeklendiriyoruz
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2]).astype("float32") / 255
#rehape yeniden boyutlandırır 
#x_train.shape
#60000 ,28,28
# x_train.shape[0] dersek 60000 çıkar indexi verdi 

x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2]).astype("float32") / 255

# Etiketleri kategorik hale çevir (0-9 arasındaki rakamları one-hot encoding yapıyoruz)
y_train = to_categorical(y_train, 10) # 10 = sınıf sayısı
y_test = to_categorical(y_test, 10)
#model oluşturma 
# %% ANN modelinin olusturulmasi ve derlenmesi
model = Sequential()

# ilk katman: 512 cell, Relu Activation function, input size 28*28=784
model.add(Dense(512, activation="relu", input_shape=(28*28,)))

# ikinci katman: 256 cell, activation: tanh
model.add(Dense(256, activation="tanh"))

# output layer: 10 tane olmak zorunda, activation softmax
model.add(Dense(10, activation="softmax"))

model.summary()

# model derlemesi: optimizer (adam: buyuk veri ve kompleks aglar icin idealdir)
# model derlemesi: loss (categorical_crossentropy)
# model derlemesi: metrik (accuracy)
model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=["accuracy"])
model.summary()
# %% Callback'lerin tanimlanmasi ve ANN egitilmesi

# Erken durdurma: val_loss iyileşmiyorsa eğitimi durduralım
# monitor: doğrulama setindeki (val) kaybı (loss) izler
# patience: 3 -> 3 epoch boyunca val loss değişmiyorsa erken durdurma yapar
# restore_best_weights: en iyi modelin ağırlıklarını geri yükler
early_stopping = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

# Model checkpoint: en iyi modelin ağırlıklarını kaydeder
# save_best_only: sadece en iyi performans gösteren modeli kaydeder
checkpoint = ModelCheckpoint("ann_best_model.h5", monitor="val_loss", save_best_only=True)

# Model eğitimi
# 60000 veri setini her biri 60 parçadan oluşan 1000 kerede train edecek ve bize 1 epoch diyecek
model.fit(x_train, y_train, #train eğitim verisi
          epochs=10,   # model toplamında 10 kere veri setini görecek yani veri seti 10 kere eğitecek
          batch_size=60, # veri setini 60 lı parçalar ile eğitim yapılacak 
          validation_split=0.2,# eğitim verilerinin %20 si doğrulama verisi olarak kullanılacak
          callbacks=[early_stopping, checkpoint])
# %% Model evaluation, gorsellestirme, model save and load
