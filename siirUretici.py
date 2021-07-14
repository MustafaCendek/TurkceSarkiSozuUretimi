from inspect import trace
from sys import exec_prefix
from typing import Sequence
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from keras.callbacks import TensorBoard
import string
import re
import os
import time
from ornek_uretme_metodlari import greedy_search, top_k_sampling, temperature_sampling

############################===AYARLAR===####################################                                                                  #
###Dizinler                                                                 #
calismaDizini = os.getcwd()                                                 #
verisetiKlasoru = os.path.join(calismaDizini, "veriseti")                   #
modelKlasoru = os.path.join(calismaDizini, "modeller")                      #
logKlasoru = os.path.join(calismaDizini, "loglar")                          #
logAdi = os.path.join(logKlasoru, "{}".format(
    time.strftime("%Y-%m-%d-%H-%M-%S")))                                    #
###Veriseti                                                                 #
verisetiAdi = "dataset.large.txt"                                             #
batchSize = 64                                                              #
girdiUzunlugu = 20                                                          #
atlama = 3                                                                  #
###Model                                                                    #
egitim = False                                                              #
essizKarakterSayisi = 160                                                   #
embeddingCiktiBoyutu = 16                                                   #
lstmCiktiBoyutu = 128                                                       #
dropOutOrani = 0.5                                                          #
lossFonksiyonu = 'sparse_categorical_crossentropy'                          #
optimizasyonYontemi = 'adam'                                                #
###Eğitim                                                                   #
epockSayisi = 5                                                             #
###Metin Uretimi                                                            #
tohum = 'Cehennem olsa gelen, göğsümüzde söndürürüz. \
    Bu yol ki hak yoldur, dönme bilmeyiz, yürürüz.'                         #
uretilenKarakterSayisi = 500                                                #
#############################################################################

f = open(os.path.join(verisetiKlasoru, verisetiAdi), "r", encoding="utf8")
veri = f.read()
print("Dosya başarıyla Okundu...")

# Verinden {girdiUzunlugu} karakteri girisi icin
# {girdiUzunlugu + 1}. karakteri de cikis icin ayıran kısım.
# {atlama} karakter otelenerek bu islem butun veri icin devam eder.
girisler = []
cikis = []
for i in range(0, len(veri) - girdiUzunlugu, atlama):
    girisler.append(veri[i: i + girdiUzunlugu])
    cikis.append(veri[i + girdiUzunlugu])
# Veriyi liste turunden tf Datasete cevirme islemi
HamGirisler = tf.data.Dataset.from_tensor_slices(girisler)
HamCikis = tf.data.Dataset.from_tensor_slices(cikis)

# Metin uzerinde islem yapan fonksiyonlar
def metinStandartlastirici(metin):
    kucukharf     = tf.strings.lower(metin)
    sayisiz  = tf.strings.regex_replace(kucukharf, "[\d-]", " ")
    isaretsiz  =tf.strings.regex_replace(sayisiz, 
                             "[%s]" % re.escape(string.punctuation), "")    
    return isaretsiz

# Metin uzerinde islem yapan fonksiyonlar
def karakterlereAyirici(metin):
    return tf.strings.unicode_split(metin, 'UTF-8')


# Katmanın Tanımlanması
vektorlestirici = TextVectorization(
    standardize=metinStandartlastirici,
    max_tokens=essizKarakterSayisi,
    split=karakterlereAyirici,
    output_mode="int",
    output_sequence_length=girdiUzunlugu)


print(vektorlestirici.get_vocabulary)
# Katmanın veriye adapte olması
vektorlestirici.adapt(HamGirisler.batch(batchSize))

# Verilen metni vektorlestiren fonksiyon
def vektorlestir(metin):
    metin = tf.expand_dims(metin, -1)
    return tf.squeeze(vektorlestirici(metin))

# Gelen vektorlesmis metini string'e ceviren fonksiyon
def stringeCevir(vektorlestirilmisMetin):
    stringDizi = []
    for token in vektorlestirilmisMetin:
        try:
            stringDizi.append(vektorlestirici.get_vocabulary()[token])
        except:
            pass
    stringMetin = ''.join(stringDizi)
    print("\t",stringMetin)
    return stringMetin


# String metinler vektorlestiriliyor
X = HamGirisler.map(vektorlestir)
Y = HamCikis.map(vektorlestir)
Y = Y.map(lambda x: x[0])
veriseti = tf.data.Dataset.zip((X, Y))

# Veriyi batchlere bölme, karıstırma
AUTOTUNE = tf.data.AUTOTUNE
veriseti = veriseti.shuffle(buffer_size=512).batch(
    batchSize, drop_remainder=True).cache().prefetch(buffer_size=AUTOTUNE)

# Modelin Örneklenmesi
# Örnekleme işlemi yapılırken Tensorflow "Functional API" yöntemi kullanılmıştır.
inputs = tf.keras.Input(shape=(girdiUzunlugu), dtype="int64")
x = layers.Embedding(essizKarakterSayisi, embeddingCiktiBoyutu)(inputs)
x = layers.Dropout(dropOutOrani)(x)
x = layers.LSTM(lstmCiktiBoyutu, return_sequences=True)(x)
x = layers.Flatten()(x)
tahminler = layers.Dense(essizKarakterSayisi, activation='softmax')(x)
model_LSTM = tf.keras.Model(inputs, tahminler, name="model_LSTM")

# Modelin Derlenmesi
model_LSTM.compile(loss=lossFonksiyonu,
                   optimizer=optimizasyonYontemi, metrics=['accuracy'])

# Modeli Kaydeden Callback
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='{0}/cp-{{epoch:02d}}-{{accuracy:.2f}}'.format(modelKlasoru),
    save_weights_only=True,
    mode='max',
    save_freq='epoch')

# Log Tutan Callback
kerasboard = TensorBoard(log_dir=logAdi,
                         batch_size=batchSize,
                         histogram_freq=1,
                         write_grads=True)

# Modelin Özetini Ekrana Yazdırma
print(model_LSTM.summary())

# Modelin Egitimi
try:
    sonModel = tf.train.latest_checkpoint(modelKlasoru)
    model_LSTM.load_weights(sonModel)
except:
    if egitim == True:
        model_LSTM.fit(veriseti, epochs=epockSayisi, callbacks=[
                       model_checkpoint_callback, kerasboard])
        egitim = False
if egitim == True:
    model_LSTM.fit(veriseti, epochs=epockSayisi,
                   callbacks=[model_checkpoint_callback, kerasboard])


# Modeli Kullanarak Metin Ureten Fonksiyon
def Uretici(model=model_LSTM, tohum=tohum, step=uretilenKarakterSayisi, yontem="greedy", parametreler=[1]):
    vektorTohum = vektorlestir(tohum)
    print("TOHUM METNI:")
    stringeCevir(vektorTohum.numpy().squeeze())
    for parametre in parametreler:
        vektorTohum = vektorlestir(tohum).numpy().reshape(1, -1)
        uretilenMetin = (vektorTohum)
        print("YONTEM: ", yontem)
        if yontem != "greedy":
            print("PARAMETRE: ", parametre)
        for i in range(step):
            tahminler = model.predict(vektorTohum)
            if yontem == "greedy":
                secim = greedy_search(tahminler.squeeze())
            elif yontem == "top-k":
                secim = top_k_sampling(tahminler.squeeze(), parametre)
            elif yontem == "temperature":
                secim = temperature_sampling(tahminler.squeeze(), parametre)
            else:
                secim = ""
            uretilenMetin = np.append(uretilenMetin, secim)
            vektorTohum = uretilenMetin[-girdiUzunlugu:].reshape(
                1, girdiUzunlugu)
        print("Üretilen Metin:")
        stringeCevir(uretilenMetin)


# Uretici Fonksiyonu Çağırma Şekilleri
Uretici(yontem="greedy")
Uretici(yontem="top-k", parametreler=[2, 3, 4, 5])
Uretici(yontem="temperature", parametreler=[0.2, 0.5, 1.0, 1.2])