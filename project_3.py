import tensorflow as tf
import numpy as np
from tensorflow import keras

# YOUR CODE
hamlet_1_text = open('text_project3/hamlet_1.txt', encoding="utf-8").read()
hamlet_2_text = open('text_project3/hamlet_2.txt', encoding="utf-8").read()
hamlet_3_text = open('text_project3/hamlet_3.txt', encoding="utf-8").read()

# YOUR CODE
tokenizer = tf.keras.preprocessing.text.Tokenizer(char_level=True)

# YOUR CODE
tokenizer.fit_on_texts([hamlet_1_text, hamlet_2_text, hamlet_3_text])

# YOUR CODE
max_id = len(tokenizer.word_index)

# YOUR CODE
hamlet_1_encoded = np.squeeze(np.array(tokenizer.texts_to_sequences(hamlet_1_text)) - 1)
hamlet_2_encoded = np.squeeze(np.array(tokenizer.texts_to_sequences(hamlet_2_text)) - 1)
hamlet_3_encoded = np.squeeze(np.array(tokenizer.texts_to_sequences(hamlet_3_text)) - 1)

# YOUR CODE
hamlet_1_dataset = tf.data.Dataset.from_tensor_slices(hamlet_1_encoded)
hamlet_2_dataset = tf.data.Dataset.from_tensor_slices(hamlet_2_encoded)
hamlet_3_dataset = tf.data.Dataset.from_tensor_slices(hamlet_3_encoded)

# YOUR CODE
T = 100
window_length = T + 1

# YOUR CODE
hamlet_1_dataset = hamlet_1_dataset.window(size = window_length, shift = 1, drop_remainder = True)
hamlet_2_dataset = hamlet_2_dataset.window(size = window_length, shift = 1, drop_remainder = True)
hamlet_3_dataset = hamlet_3_dataset.window(size = window_length, shift = 1, drop_remainder = True)

# YOUR CODE
hamlet_1_dataset = hamlet_1_dataset.flat_map(lambda window: window.batch(window_length))
hamlet_2_dataset = hamlet_2_dataset.flat_map(lambda window: window.batch(window_length))
hamlet_3_dataset = hamlet_3_dataset.flat_map(lambda window: window.batch(window_length))

# YOUR CODE
hamlet_dataset = hamlet_1_dataset.concatenate(hamlet_2_dataset)
hamlet_dataset = hamlet_dataset.concatenate(hamlet_3_dataset)

tf.random.set_seed(0)
# YOUR CODE
batch_size = 32
hamlet_dataset = hamlet_dataset.repeat()
hamlet_dataset = hamlet_dataset.shuffle(buffer_size = 10000)
hamlet_dataset = hamlet_dataset.batch(batch_size, drop_remainder=True)

hamlet_dataset = hamlet_dataset.map(lambda window_batch: (window_batch[:, 0:100], window_batch[:, 1:101]))

# YOUR CODE
hamlet_dataset = hamlet_dataset.map(lambda X_batch, Y_batch: (tf.one_hot(X_batch, max_id), Y_batch))

# YOUR CODE
hamlet_dataset = hamlet_dataset.prefetch(buffer_size = 1)

# YOUR CODE
steps_per_epoch= int(((len(hamlet_1_encoded)+len(hamlet_2_encoded)+len(hamlet_3_encoded))-3*T)/batch_size)

# YOUR CODE
model = keras.models.Sequential()
model.add(keras.layers.GRU(128, return_sequences = True, input_shape=[None, max_id]))
model.add(keras.layers.GRU(128, return_sequences = True))
model.add(keras.layers.TimeDistributed(keras.layers.Dense(max_id,activation="softmax")))

# YOUR CODE
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(),
    optimizer=keras.optimizers.Adam(),
    metrics=["accuracy"],
)

# YOUR CODE
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
history = model.fit(hamlet_dataset, epochs=32, steps_per_epoch=steps_per_epoch, callbacks=[callback])

model.save('hamlet_model.h5')