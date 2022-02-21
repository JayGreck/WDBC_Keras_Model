from Preprocess_Data import Preprocess_Data
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy


preprocess = Preprocess_Data()

X_train, X_test, y_train, y_test = preprocess.get_dataframe()

# ------- SELECTING GPU INSTEAD OF CPU FOR COMPUTATIONS -------
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(physical_devices))
if len(physical_devices) != 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# ------- TRAIN NEURAL NETWORK -------
model = Sequential([
    Dense(units = 5, input_shape=(0,30), activation='sigmoid'),
    Dense(units = 5, activation='sigmoid'),
    Dense(units = 5, activation='sigmoid'),
    Dense(units = 2, activation='sigmoid')
])
#print(model.summary())

# Preparing the model for training
model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the Model
model.fit(x=X_train, y=X_test, batch_size=10, epochs=2000, verbose=2)