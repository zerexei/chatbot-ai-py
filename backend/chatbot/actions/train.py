from django.conf import settings
import logging
import os
import json
import time
import random
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
import contractions


class BotModel:
    def __init__(self, epochs=300, batch_size=64):
        self.logger = logging.getLogger(__name__)
        self.epochs = epochs
        self.batch_size = batch_size

    def train(self):
        intents = self._get_intents()
        words, classes, documents = self._process_intents(intents)
        words = self._lemmatize(words)
        
        words = sorted(list(set(words)))
        with open('words.pkl', 'wb') as handle:
            pickle.dump(words, handle, protocol=pickle.HIGHEST_PROTOCOL)

        classes = sorted(list(set(classes)))
        with open('classes.pkl', 'wb') as ecn_file:
            pickle.dump(classes, ecn_file, protocol=pickle.HIGHEST_PROTOCOL)

        train_x, train_y, input_size, output_size = self._create_training_data(words, classes, documents)
        self._train(train_x, train_y,input_size, output_size)
        
       

    def getProps(self):
        return f"epochs: {self.epochs} - batch size: {self.batch_size}"

    def _get_intents(self):
        intents_path = os.path.join(settings.BASE_DIR, 'chatbot/intents.json')

        try:
            with open(intents_path, 'r') as file:
                content = json.load(file)
            return content
        except (FileNotFoundError, IOError):
            self.logger.error('File not found.')

    def _process_intents(self, intents):
        words = []
        classes = []
        documents = []

        for intent in intents:
            for pattern in intent['patterns']:
                try:
                     pattern = contractions.fix(pattern)
                     w = nltk.word_tokenize(pattern)
                     words.extend(w)
                     documents.append((w, intent['tag']))

                     if intent['tag'] not in classes:
                         classes.append(intent['tag'])

                except Exception as e:
                    self.logger(f"Error processing pattern: {pattern}. {e}")
        
        return words, classes, documents

    # TODO: refactor: bag, matrix variable name
    def _create_training_data(self, words, classes, documents):
        training = []
        matrix = [0] * len(classes)


        for doc in documents:
            bag = []
            pattern_words = self._lemmatize(doc[0])

            for w in words:
                bag.append(1) if w in pattern_words else bag.append(0)

            matrix[classes.index(doc[1])] = 1
            training.append([bag, matrix])

        max_length = max(len(bag) for bag, _ in training)
        
        for i, (bag, matrix) in enumerate(training):
            padding = [0] * (max_length - len(bag))
            training[i] = (bag + padding, matrix)

        random.shuffle(training)

        training = np.array(training, dtype=object)

        train_x = list(training[:, 0]) # [bag, bag, ...]
        train_y = list(training[:, 1]) # [matrix, matrix, ...]

        input_size = len(train_x[0])
        output_size = len(train_y[0])

        return train_x, train_y, input_size, output_size
    
    def _train(self, train_x, train_y, input_size, output_size):
        model = Sequential()

        model.add(Dense(128, input_shape=(input_size,), activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(128, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(output_size, activation='softmax'))

        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        start_time = time.time()
        
        history = model.fit(
            np.array(train_x), 
            np.array(train_y),
            epochs=self.epochs, 
            batch_size=self.batch_size, 
            verbose=True
        )
        
        model.save('chatbot_model.h5', history)

        end_time = time.time()

        # Analytics
        model.summary()
        training_time = end_time - start_time
        loss = history.history["loss"][-1]
        accuracy = history.history["accuracy"][-1]

        print("Training time: {:.2f} seconds".format(training_time))
        print(f"Final loss: {loss:.4f}", "|", f"Final accuracy: {accuracy:.4f}")
    
    def _lemmatize(self, words):
        lemmatizer = WordNetLemmatizer()

        ignore_words = ['\u200d', '?', '....', '..', '...', '', '@', '#', ',', '.', '"', ':', ')', '(', '-', '!', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '*', '+', '\\','•', '~', '£', '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█','½', 'à', '…', '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓','—', '‹', '─', '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾','Ã', '⋅', '‘', '∞', '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø','¹', '≤', '‡', '√', '!', '🅰', '🅱']
        
        words = [lemmatizer.lemmatize(w.lower())
             for w in words if w not in ignore_words]

        return words

