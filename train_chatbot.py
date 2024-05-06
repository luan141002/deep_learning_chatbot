#%%
import nltk
import json
import pickle
import random
import numpy as np
from nltk.stem import WordNetLemmatizer
from keras import models, layers, optimizers
from keras.callbacks import EarlyStopping

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('wordnet')

# Function to preprocess data
def preprocess_data():
    words = []
    classes = []
    documents = []
    ignore_words = ['?', '!', '@', '$']

    with open('intents.json', 'r', encoding='utf-8') as data_file:
        intents = json.load(data_file)

    lemmatizer = WordNetLemmatizer()

    for intent in intents['intents']:
        for pattern in intent['patterns']:
            if isinstance(pattern, str):
                w = nltk.word_tokenize(pattern)
                
                words.extend(w)
                documents.append((w, intent['tag']))

                if intent['tag'] not in classes:
                    classes.append(intent['tag'])
            else:
                print("Skipping non-string pattern:", pattern)
    words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
    words = sorted(list(set(words)))
    classes = sorted(list(set(classes)))
    

    print(len(words))

    pickle.dump(words, open('words.pkl', 'wb'))
    pickle.dump(classes, open('classes.pkl', 'wb'))

    training_data = []
    output_empty = [0] * len(classes)

    for doc in documents:
        bag = []
        pattern_words = doc[0]
        pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]

        for w in words:
            bag.append(1) if w in pattern_words else bag.append(0)

        output_row = list(output_empty) 
        output_row[classes.index(doc[1])] = 1

        training_data.append([bag, output_row])
    
    random.shuffle(training_data)
    training_data = np.asarray(training_data, dtype="object")
    train_x = list(training_data[:, 0])
    train_y = list(training_data[:, 1])

    return train_x, train_y, classes, words

# Function to create and train model
def create_and_train_model(train_x, train_y, words):
    model = models.Sequential([
        layers.Dense(len(words), input_shape=(len(train_x[0]),), activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1024, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(256 , activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),  
        layers.Dropout(0.3),
        layers.Dense(len(train_y[0]), activation='softmax')
    ])

    model.compile(loss="categorical_crossentropy",
                  optimizer=optimizers.Adam(learning_rate=0.001),
                  metrics=["accuracy"])

    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    hist = model.fit(np.array(train_x), np.array(train_y),
                     epochs=250,
                     batch_size=24,
                     verbose=1,
                     callbacks=[early_stopping])

    model.save('chatbot_model_5.h5', hist)

# Function to evaluate model
def evaluate_model(model, test_x, test_y):
    loss, accuracy = model.evaluate(np.array(test_x), np.array(test_y))
    print("Loss:", loss)
    print("Accuracy:", accuracy)

# Main function
def main():
    train_x, train_y, classes,words = preprocess_data()
  
    # Splitting data into test and train
    test_size = int(len(train_x) * 0.2)
    test_x = train_x[-test_size:]
    test_y = train_y[-test_size:]
    train_x = train_x[:-test_size]
    train_y = train_y[:-test_size]

    create_and_train_model(train_x, train_y,words)
    model = models.load_model('chatbot_model_5.h5')

    evaluate_model(model, test_x, test_y)


main()
