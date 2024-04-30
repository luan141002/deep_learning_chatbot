
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

def load_intents(filename):
    """Load intents from a JSON file."""
    with open(filename, 'r', encoding='utf-8') as data_file:
        intents = json.load(data_file)
    return intents

def preprocess_data(intents, lemmatizer, ignore_words):
    """Preprocess intents data."""
    words = []
    classes = []
    documents = []

    for intent in intents['intents']:
        for pattern in intent['patterns']:
            w = nltk.word_tokenize(pattern)
            words.extend(w)
            documents.append((w, intent['tag']))

            if intent['tag'] not in classes:
                classes.append(intent['tag'])

    words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
    words = sorted(list(set(words)))
    classes = sorted(list(set(classes)))

    return words, classes, documents

def create_training_data(documents, words, classes, lemmatizer):
    """Create training data."""
    training = []
    output_empty = [0] * len(classes)

    for doc in documents:
        bag = []
        pattern_words = doc[0]
        pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]

        for w in words:
            bag.append(1) if w in pattern_words else bag.append(0)

        output_row = list(output_empty)
        output_row[classes.index(doc[1])] = 1

        training.append([bag, output_row])

    random.shuffle(training)
    training = np.asarray(training, dtype="object")
    train_x = list(training[:,0])
    train_y = list(training[:,1])

    return train_x, train_y

def split_data(train_x, train_y, test_size=0.2):
    """Split data into training and testing sets."""
    test_size = int(len(train_x) * test_size)

    test_x = train_x[-test_size:]
    test_y = train_y[-test_size:]

    train_x = train_x[:-test_size]
    train_y = train_y[:-test_size]

    return train_x, train_y, test_x, test_y

def save_data(words, classes, wordsfile, classfile):
    """Save words and classes to pickle files."""
    pickle.dump(words, open(wordsfile, 'wb'))
    pickle.dump(classes, open(classfile, 'wb'))


intents = load_intents('intents.json')
lemmatizer = nltk.stem.WordNetLemmatizer()
ignore_words = ['?', '!', '@', '$']

words, classes, documents = preprocess_data(intents, lemmatizer, ignore_words)
save_data(words, classes,'words.pkl','classes.pkl')

train_x, train_y = create_training_data(documents, words, classes, lemmatizer)
train_x, train_y, test_x, test_y = split_data(train_x, train_y)



#%% Create model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
# equal to number of intents to predict output intent with softmax

model = models.Sequential([
    layers.Dense(512, input_shape=(len(train_x[0]),), activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(384, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),  # Additional layer
    layers.Dropout(0.5),
    layers.Dense(len(train_y[0]), activation='softmax')
])
#%% Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model

learning_rate = 0.001
momentum = 0.9
nesterov = True
# optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum, nesterov=nesterov)
model.compile(loss = "categorical_crossentropy",
              optimizer = optimizers.Adam(learning_rate=0.001), #Stochastic Gradient Descent
              metrics = ["accuracy"])

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)              
#%% fitting and saving the model
hist = model.fit(
    np.array(train_x), np.array(train_y),
      epochs=200,
        batch_size=16,
          verbose=1,
          callbacks=[early_stopping])
model.save('chatbot_model_4.h5', hist)
# Evaluate model
loss, accuracy = model.evaluate(np.array(test_x), np.array(test_y))
print("Loss:", loss)
print("Accuracy:", accuracy)
print("model created")

 #%%
from tensorflow.keras.models import load_model
# Load the pre-trained model
chatbot_model = load_model('chatbot_model_4.h5')

intents = load_intents('intents_gym.json')
lemmatizer = nltk.stem.WordNetLemmatizer()
ignore_words = ['?', '!', '@', '$']

words, classes, documents = preprocess_data(intents, lemmatizer, ignore_words)
save_data(words, classes,'words_1.pkl','classes_1.pkl')
new_train_x, new_train_y = create_training_data(documents, words, classes, lemmatizer)
new_train_x, new_train_y, new_test_x, new_test_y = split_data(train_x, train_y)

chatbot_model = Sequential([
    layers.Dense(256, input_shape=(len(train_x[0]),), activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),  # Additional layer
    layers.Dropout(0.5),
    layers.Dense(len(train_y[0]), activation='softmax')
])

chatbot_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Fine-tuning: Continue training on new data
fine_tuning_hist = chatbot_model.fit(new_train_x, new_train_y, epochs=250, batch_size=24, verbose=1, callbacks=[early_stopping])

# Save the fine-tuned model
chatbot_model.save('fine_tuned_chatbot_model.h5')

loss, accuracy = chatbot_model.evaluate(np.array(new_test_x), np.array(new_test_y))
print("Loss:", loss)
print("Accuracy:", accuracy)

# %%
