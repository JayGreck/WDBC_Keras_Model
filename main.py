from Preprocess_Data import Preprocess_Data
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, precision_score, mean_squared_error, accuracy_score
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


preprocess = Preprocess_Data()

X_train, X_test, y_train, y_test = preprocess.get_dataframe()

# ------- SELECTING GPU INSTEAD OF CPU FOR COMPUTATIONS -------
# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# print("Num GPUs Available: ", len(physical_devices))
# if len(physical_devices) != 0:
#     tf.config.experimental.set_memory_growth(physical_devices[0], True)

# ------- CREATE NEURAL NETWORK MODEL -------
model = Sequential([
    Dense(units = 5, input_shape=(0,30), activation='sigmoid'),
    Dense(units = 5, activation='sigmoid'),
    Dense(units = 5, activation='sigmoid'),
    Dense(units = 2, activation='sigmoid')
])
#print(model.summary())


# ------- PREPEARE MODEL -------
model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy', tf.keras.metrics.MeanSquaredError()])

# ------- TRAIN/ VALIDATION NEURAL NETWORK -------
test = model.fit(x=X_train, y=X_test, validation_split=0.3, batch_size=10, epochs=2000, shuffle=True, verbose=2)


# ------- TEST THE MODEL -------
predictions = model.predict(x=y_train, batch_size=10, verbose=0)
rounded_predictions = np.argmax(predictions, axis=-1)

# ------- CONFUSION MATRIX -------
cm = confusion_matrix(y_true=y_test, y_pred=rounded_predictions)

TN = cm[0][0]
FN = cm[1][0]
TP = cm[1][1]
FP = cm[0][1]

# ------- METRIC SCORES -------
TPR = TP / (TP+FN)
FPR = FP / (FP + TN)
TP = precision_score(y_test, rounded_predictions, average='binary')
MSE = mean_squared_error(y_test, rounded_predictions)
accuracy = accuracy_score(y_test, rounded_predictions, normalize=True)
print(MSE)
print(TP)
print("Mean Squared Error " + str(MSE))
print("FPR: " + str(FPR))
print("TPR: " + str(TPR))
print("Accuracy: " + str(accuracy) + " %")

# ------- CONFUSION MATRIX VISUALISATION -------
classes = ['Benign', 'Malignant']
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
plt.show()
