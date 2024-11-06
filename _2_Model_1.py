################## LSTM with attention
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
import os
import _1_Data_Extraction 


current_directory = os.path.dirname(os.path.abspath(__file__))
# Ensure TensorFlow logs are suppressed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Step 1: Extract text data
text = _1_Data_Extraction.Helper1()
print('2. Extraction completed moving on to pre-processing....')

########## STEP-2: Data Pre-Processing ##########

# SUB STEP-2.1: Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
total_words = len(tokenizer.word_index) + 1

# SUB STEP-2.2: Create sequences of words
input_sequences = []
for line in text.split('\n'):
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

# SUB STEP-2.3: Pad sequences to ensure equal length
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

# SUB STEP-2.4: Split the data into input (X) and output (y)
X = input_sequences[:,:-1]
y = input_sequences[:,-1]
y = tf.keras.utils.to_categorical(y, num_classes=total_words)  # One-hot encoding

print('3. Task completed, terminating the processes....')

# Step 3: Define the Model
model = Sequential()
model.add(Embedding(input_dim=total_words, output_dim=100))
model.add(LSTM(150, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dense(total_words, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# View the model summary
model.summary()

# Step 4: Train the Model
history = model.fit(X, y, epochs=2, verbose=1)

# Step 5: Save the Model
model_path = os.path.join(current_directory, "_1_lstm.h5")
tokenizer_path= os.path.join(current_directory, '_1_tokenizer.pkl')
max_seq_path= os.path.join(current_directory, '_1_max_seq_len.pkl')
model.save(model_path)
# Saving the tokenizer
with open(tokenizer_path, 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
# Saving the max_sequence_len
with open(max_seq_path, 'wb') as handle:
    pickle.dump(max_sequence_len, handle, protocol=pickle.HIGHEST_PROTOCOL)

print(f"Model saved to {model_path}")
