import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# MNIST podatkovni skup
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# TODO: prikazi nekoliko slika iz train skupa
plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_train[i], cmap="gray")
    plt.title(f"BROJ: {y_train[i]}")
    plt.axis("off")
plt.tight_layout()
plt.show()


# Skaliranje vrijednosti piksela na raspon [0,1]
x_train_s = x_train.astype("float32") / 255
x_test_s = x_test.astype("float32") / 255

# Slike 28x28 piksela se predstavljaju vektorom od 784 elementa
x_train_s = x_train_s.reshape(60000, 784)
x_test_s = x_test_s.reshape(10000, 784)

# Kodiraj labele (0, 1, ... 9) one hot encoding-om
y_train_s = keras.utils.to_categorical(y_train, 10)
y_test_s = keras.utils.to_categorical(y_test, 10)


# TODO: kreiraj mrezu pomocu keras.Sequential(); prikazi njenu strukturu pomocu .summary()
model = keras.Sequential([
layers.Dense(128, activation="relu", input_shape=(784,)),
layers.Dense(64, activation="relu"),
layers.Dense(10, activation="softmax")
])


model.summary()



# TODO: definiraj karakteristike procesa ucenja pomocu .compile()

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])


# TODO: provedi treniranje mreze pomocu .fit()
history = model.fit(x_train_s, y_train_s,epochs=4,batch_size=32,validation_split=0.2)


# TODO: Izracunajte tocnost mreze na skupu podataka za ucenje i skupu podataka za testiranje
train_loss, train_acc = model.evaluate(x_train_s, y_train_s, verbose=0)
test_loss, test_acc = model.evaluate(x_test_s, y_test_s, verbose=0)
print(f"\nTrain Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}")


# TODO: Prikazite matricu zabune na skupu podataka za testiranje

y_test_pred = model.predict(x_test_s)
y_test_pred_classes = np.argmax(y_test_pred, axis=1)

cm = confusion_matrix(y_test, y_test_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.arange(10), yticklabels=np.arange(10))
plt.title("Matrica")
plt.xlabel('Predviden')
plt.ylabel('Stvarni')
plt.show()