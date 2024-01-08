from tkinter import *
import tkinter as tk
import tensorflow as tf
import tkinter.filedialog as filedialog
import numpy as np
from PIL import Image, ImageTk

# Kullanıcı arayüzünü oluştur
root = tk.Tk()
root.title("Nesne Tanıma")
root.geometry("800x600")

title = tk.Label(root, text="Nesne Tanıma uygulamasını kullanmak için resim seç butonuna tıklayınız.",background="light blue")
title.config(font=("Arial",10))
title.pack

#Resim dosyası seçme ve yükleme işlemi için bir düğme
def select_image():
  #Resim dosyasını seç
  file_path = filedialog.askopenfilename()

#Seçilen resmi oku ve bir PhotoImage objesine dönüştür
image = Image.open(file_path)
image = ImageTk.PhotoImage(image)

#Seçilen resmi arayüze yaz
image_label = tk.Label(image=image)
image_label.image = image
image_label.configure(width=2000, height=500)
image_label.pack()

#Resim dosyasını okuma ve bir numpy dizisine dönüştürme
image = tf.io.read_file(file_path)
image =tf.image.decode_jpeg(image, channels=3)

#Girdi verilerini tanımla
inputs = tf.keras.Input(shape=(299,299,3))

#Sinir ağı modelini oluştur
x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3),activation='relu')(inputs)
x = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(x)
x= tf.keras.layers.Flatten()(x)
x= tf.keras.layers.Dense(units=64,activation ='relu')(x)
outputs = tf.keras.layers.Dense(units=10, activation='softmax')(x)

#Modeli derle
model= tf.keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

 #Resmi işle özniteliklerini bul
 image = tf.image.resize(image, (299.299))
 image = tf.keras.applications.inception_v3.preprocess_input(image)

 #Öznitelikleri kullanarak nesne türünü tahmin etme
 prediction = model.predict(image[tf.newaxis, ...])
 prediction = tf.argmax(prediction, axis=-1)
 prediction=string =int(prediction)

 #Girdi verilerini numpy dizisine dönüştür
 input_data = np.expand_dims(image, axis=0)

 #Modeli kullanarak tahmini yap
 predictions = model.predict(input_data)

 #Tahmin edilen sınıf indeksini bulun
 prediction = np.argmax(predictions)

 #Nesne türlerine karşılık gelen dizeleri tanımla
 classes = ["apple","banana","berry","coconut","fig","grape","orange","pear"]

 #Tahmin edilen nesne türünü bir dize olarak döndürmek için
 prediction_string =classes[prediction]

 title2 = tk.Label(root, text="Tahminim", background="light blue")
 title2.config(font=("Arial",10))
 title2.pack()

 #Tahmin edilen nesne türünü etiket olarak gösterme
 label = tk.Label(root, text=rediction_string)
 label.pack()

button = tk.Button(text="Çıkış", width=4, height=2, bg="white", fg="blue", commad=root.destroy)
button2.pack(side=tk.BOTTOM)

root.mainloop()
