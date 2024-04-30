
#%% import libraries
import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle
from keras import regularizers, models, layers, optimizers
from keras.callbacks import EarlyStopping
import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, LSTM, LeakyReLU
from keras.optimizers import SGD
import random
#%% Declare essential variables and import datasets
words=[]
classes = []
documents = []
ignore_words = ['?', '!', '@', '$']
# Open the file with explicit encoding
with open('intents.json', 'r', encoding='utf-8') as data_file:
    intents = json.load(data_file)

#%% Language Handling and labeling all word
for intent in intents['intents']:
    for pattern in intent['patterns']:

        # take each word and tokenize it
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        # adding documents
        documents.append((w, intent['tag']))

        # adding classes to our class list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

classes = sorted(list(set(classes)))

print(len(documents), "documents", documents)

print(len(classes), "classes", classes)

print(len(words), "unique lemmatized words", words)


pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))

#%% initializing training data
training = []
output_empty = [0] * len(classes)
for doc in documents:
    # initializing bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # lemmatize each word - create base word, in attempt to represent related words
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    # create our bag of words array with 1, if word match found in current pattern
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    # output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])
# shuffle our features and turn into np.array
random.shuffle(training)
training =  np.asarray(training, dtype="object")
print(training)
train_x = list(training[:,0])
train_y = list(training[:,1])


# Test Data (20%)
test_size = int(len(train_x) * 0.2)

# Divide data into test and train data
test_x = train_x[-test_size:]
test_y = train_y[-test_size:]

# Get train data
train_x = train_x[:-test_size]
train_y = train_y[:-test_size]
print("Training data created")


#%% Create model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
# equal to number of intents to predict output intent with softmax

# model = models.Sequential([
#     layers.Dense(64, input_shape=(len(train_x[0]),), activation='relu'),
#     layers.Dropout(0.5),
#     layers.Dense(32 , activation='relu'),
#     layers.Dropout(0.5),
#     # layers.Dense(256, activation='relu'),
#     # layers.Dropout(0.3),
#     # layers.Dense(128, activation='relu'),  # Additional layer
#     # layers.Dropout(0.3),
#     layers.Dense(len(train_y[0]), activation='softmax')
# ])

# model = models.Sequential([
#     layers.Dense(512, input_shape=(len(train_x[0]),), activation='relu'),
#     layers.Dropout(0.5),
#     layers.Dense(384, activation='relu'),
#     layers.Dropout(0.5),
#     layers.Dense(256, activation='relu'),
#     layers.Dropout(0.5),
#     layers.Dense(128, activation='relu'),  # Additional layer
#     layers.Dense(len(train_y[0]), activation='softmax')
# ])
#%% Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
model = models.Sequential([
    layers.LSTM(64, input_shape=(len(train_x[0]), 1), return_sequences=True),
    layers.Dropout(0.5),
    layers.LSTM(32),
    layers.Dropout(0.5),
    layers.Dense(len(train_y[0]), activation='softmax')
])
# optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum, nesterov=nesterov)
model.compile(loss = "categorical_crossentropy",
              optimizer = optimizers.Adam(learning_rate=1e-3), #Stochastic Gradient Descent
              metrics = ["accuracy"])

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)              
#%% fitting and saving the model
hist = model.fit(
    np.array(train_x), np.array(train_y),
      epochs=250,
        batch_size=24,
          verbose=1,
          callbacks=[early_stopping])
model.save('chatbot_model_5.h5', hist)
#%% Evaluate model
loss, accuracy = model.evaluate(np.array(test_x), np.array(test_y))
print("Loss:", loss)
print("Accuracy:", accuracy)
print("model created")
#%%
# from tensorflow.keras.models import load_model
# Load the pre-trained model
# chatbot_model = load_model('chatbot_model_5.h5')
# new_train_x = train_x[:-test_size]
# new_train_y = train_y[:-test_size]

# # chatbot_model = Sequential([
# #     layers.Dense(512, input_shape=(len(train_x[0]),), activation='relu'),
# #     layers.Dropout(0.5),
# #     layers.Dense(384, activation='relu'),
# #     layers.Dropout(0.5),
# #     layers.Dense(256, activation='relu'),
# #     layers.Dropout(0.5),
# #     layers.Dense(128, activation='relu'),  # Additional layer
# #     layers.Dropout(0.5),
# #     layers.Dense(len(train_y[0]), activation='softmax')
# # ])
# # chatbot_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# # Fine-tuning: Continue training on new data
# fine_tuning_hist = chatbot_model.fit(new_train_x, new_train_y, epochs=50, batch_size=5, verbose=1)

# # Save the fine-tuned model
# chatbot_model.save('fine_tuned_chatbot_model.h5')

# # %%
# loss, accuracy = chatbot_model.evaluate(np.array(test_x), np.array(test_y))
# print("Loss:", loss)
# print("Accuracy:", accuracy)

# # %%

# %%
