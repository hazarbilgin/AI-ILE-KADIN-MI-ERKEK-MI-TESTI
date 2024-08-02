# Gerekli kütüphaneleri import ediyoruz
#author=hazarbilgin
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from keras import utils, models, layers
from keras.utils import to_categorical
import keras
from keras import layers, models, preprocessing, Sequential
import shutil
from PIL import Image

# Veri seti yollarını belirliyoruz
path = 'C:\\Users\\Hazar\\Dataset'
val_path = 'C:\\Users\\Hazar\\Dataset'

# Görüntüleri ve etiketleri yükleyen işlev
def load_images_and_labels(directory, label):
    images = []
    labels = []
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            img_path = os.path.join(directory, filename)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (128, 128)) # Tutarlılık için yeniden boyutlandır
            images.append(img)
            labels.append(label)
    return images, labels

# Erkek ve kadın klasörlerini belirliyoruz
men_dir = os.path.join(path, 'MEN')
women_dir = os.path.join(path, 'WOMAN')

# Görüntüleri yüklüyoruz
men_images, men_labels = load_images_and_labels(men_dir, 0)
women_images, women_labels = load_images_and_labels(women_dir, 1)

# Verileri birleştiriyoruz
images = np.array(men_images + women_images)
labels = np.array(men_labels + women_labels)

# Etiketleri one-hot encode yapıyoruz
labels = to_categorical(labels, num_classes=2)

# Veriyi eğitim ve test seti olarak bölüyoruz
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Görüntü boyutları ve batch size belirliyoruz
img_size = (128, 128)
batch_size = 32
img_height = 128
img_width = 128

# Bozuk resimleri kontrol eden ve silen işlev
def remove_corrupt_images(directory):
    num_skipped = 0
    for folder_name in ("MEN", "WOMAN"):
        folder_path = os.path.join(directory, folder_name)
        for fname in os.listdir(folder_path):
            fpath = os.path.join(folder_path, fname)
            try:
                # Resmi aç ve kanalları kontrol et
                with Image.open(fpath) as img:
                    img.verify()  # Resim bozuksa burada hata verir
                    if img.mode not in ('RGB', 'L'):  # Kanal kontrolü
                        raise ValueError(f"Invalid image mode: {img.mode}")
            except (IOError, SyntaxError, ValueError) as e:
                print(f"Siliniyor: {fpath} ({e})")
                num_skipped += 1
                os.remove(fpath)
    print(f"Toplamda {num_skipped} resim silindi.")

# Bozuk resimleri temizliyoruz
remove_corrupt_images(path)

# Eğitim ve doğrulama veri setlerini oluşturuyoruz
train_ds = keras.preprocessing.image_dataset_from_directory(
    path,
    image_size=img_size,
    batch_size=batch_size,
    label_mode='int',
    validation_split=0.2,
    subset='training',
    seed=123
)
val_ds = keras.preprocessing.image_dataset_from_directory(
    val_path,
    image_size=img_size,
    batch_size=batch_size,
    label_mode='int',
    validation_split=0.2,
    subset='validation',
    seed=123
)

# Sınıf isimlerini alıyoruz
class_names = train_ds.class_names
print(class_names)

# Normalizasyon katmanı ekliyoruz
normalization_layer = layers.Rescaling(1./255)
normalized_train_dataset = train_ds.map(lambda x, y: (normalization_layer(x), y))
normalized_validation_dataset = val_ds.map(lambda x, y: (normalization_layer(x), y))

# Eğitim veri setinden bir batch görüntü alıp gösteriyoruz
image_batch, label_batch = next(iter(train_ds))

plt.figure(figsize=(10, 10))
for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(image_batch[i].numpy().astype("uint8"))
    label = label_batch[i]
    plt.title(class_names[label])
    plt.axis("off")

# Modeli tanımlıyoruz
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D(),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),  # Giriş boyutunu düzleştirir
    layers.Dense(512, activation='relu'),  # Örnek bir dense katman
    layers.Dense(2, activation='softmax')  # Çıkış katmanı
])

# Model özetini gösteriyoruz
model.summary()

# Modeli derliyoruz
model.compile(
    optimizer='adam',
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

# Modeli eğitiyoruz
model.fit(
    normalized_train_dataset,
    validation_data=normalized_validation_dataset,
    epochs=10
)

# Modeli kaydediyoruz
model.save("manorwomen.h5")

# Görüntüleri ve etiketleri gösteren işlev
def show_image(image, label_batch):
    plt.figure()
    plt.imshow(image)
    plt.title(label_batch.numpy())
    plt.axis('off')
    plt.show()

# Eğitim veri setinden ilk 5 görüntüyü gösteriyoruz
for images, label_batch in train_ds.take(1):
    for i in range(5):  # İlk 5 görüntü
        show_image(images[i], label_batch[i])

# Yeni bir resmi sınıflandırıp ilgili klasöre taşıyan işlev
def classify_and_move_image(image_path, model, class_names, base_path):
    Img = Image.open(image_path)
    Img = Img.resize((img_width, img_height), Image.Resampling.LANCZOS)
    img_array = np.array(Img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    predictions = model.predict(img_array)
    predict_class = class_names[np.argmax(predictions)]
    
    # Yeni dosya yolunu oluştur
    new_file_path = os.path.join(base_path, predict_class, os.path.basename(image_path))
    shutil.move(image_path, new_file_path)
    print(f"{os.path.basename(image_path)} dosyası {predict_class} klasörüne taşındı.")

# Yeni bir görüntüyü sınıflandırıp taşıyoruz
new_image_path = "C:\\Users\\Hazar\\yenidataset\\0047.jpg"
classify_and_move_image(new_image_path, model, class_names, "C:\\Users\\Hazar\\dataset")

# Modeli değerlendiriyoruz
# loss, accuracy = model.evaluate(X_test, y_test)
# print(f"Test Accuracy: {accuracy}")

# Sınıflandırma raporu ve karmaşıklık matrisi oluşturuyoruz
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Örnek tahminleri gösteren işlev
def plot_sample_predictions(X, y_true, y_pred, num_samples=10):
    indices = np.random.choice(range(len(X)), num_samples, replace=False)
    plt.figure(figsize=(20, 4))
    for i, idx in enumerate(indices):
        plt.subplot(2, num_samples // 2, i + 1)
        plt.imshow(X[idx])
        plt.title(f"True: {np.argmax(y_true[idx])}, Pred: {np.argmax(y_pred[idx])}")
        plt.axis('off')
    plt.show()

# Örnek tahminleri gösteriyoruz
plot_sample_predictions(X_test, y_test, y_pred)
#author=hazarbilgin
