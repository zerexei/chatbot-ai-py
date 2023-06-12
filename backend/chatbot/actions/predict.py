import random
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
import contractions

# TODO: create Bot model, remove redundancy
# TODO: create try/except
class BotPredict:
    def __init__(self):
        pass

    def predict(self, sentence):
        predicted_tag = self.predict_tag(sentence)
        return self.get_response(predicted_tag)
    
    def predict_tag(self, sentence):
        word_vector = self._create_word_vector(sentence)
    
        model = pickle.load(open('model.pkl', 'rb'))
        accuracy_list = model.predict(word_vector)[0]

        ERROR_THRESHOLD = 0.75
        # Return on failing accuracy
        if max(accuracy_list) < ERROR_THRESHOLD:
            return {"tag": "None", "probability": "None"}
    
        # Filter and sort results based on the accuracy threshold
        high_probable_list = [
            (index, accuracy)
            for index, accuracy in enumerate(accuracy_list)
            if accuracy > ERROR_THRESHOLD
        ]

        high_probable_list.sort(key=lambda x: x[1], reverse=True)

         # load tags encoder object
        with open('classes.pkl', 'rb') as ecn_file:
            classes = pickle.load(ecn_file)

        # Retrieve the highest probable intent and its probability
        highest_intent_index, highest_intent_accuracy = high_probable_list[0]
        highest_intent = classes[highest_intent_index]
        highest_intent_probability = str(highest_intent_accuracy)

        return {"tag": highest_intent, "probability": highest_intent_probability}

    def get_response(self, predicted_tag):
        intents = pickle.load(open('intents.pkl', 'rb'))
        intent_responses = [i['responses'] for i in intents if i['tag'] == predicted_tag['tag']]
        return random.choice(intent_responses[0]) if intent_responses else None
        
    def _create_word_vector(self, sentence):
        sentence_words = self._lemmatize(sentence)

        with open('words.pkl', 'rb') as handle:
            pickle_words = pickle.load(handle)

        return [
            [1 if word in sentence_words else 0 for word in pickle_words]
        ]

    def _lemmatize(self, sentence):
        sentence = contractions.fix(sentence)

        if not isinstance(sentence, str):
            raise Exception("Variable 'sentence' should be of type str")

        sentence_words = nltk.word_tokenize(sentence.lower())
        lemmatizer = WordNetLemmatizer()
        return [lemmatizer.lemmatize(word) for word in sentence_words]