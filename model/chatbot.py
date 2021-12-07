import json
import numpy as np
from tensorflow.keras.models import load_model


class ChatBot:
    __instance = None

    @staticmethod
    def getInstance():
        """ Static access method. """
        if ChatBot.__instance is None:
            ChatBot()
        return ChatBot.__instance

    def __init__(self):
        """ Virtually private constructor. """
        if ChatBot.__instance is not None:
            raise Exception("This class is a singleton!")

        ChatBot.__instance = self
        model_path = './model/chatbot_model.h5'
        intent_path = './model/intents.json'
        word_path = './model/words.json'
        classes_path = './model/classes.json'

        with open(intent_path, "r") as f:
            data = json.load(f)
            self.intents = data["intents"]
        with open(word_path, "r") as f:
            self.words = json.load(f)
        with open(classes_path, "r") as f2:
            self.classes = json.load(f2)

        self.model = load_model(model_path)

        self.ignore_letters = ["?", ",", ".", "!"]

    def preprocess(self, sentence):
        # word_list = nltk.word_tokenize(sentence)
        word_list = sentence.strip().split(" ")
        word_list = [word.lower() for word in word_list
                     if word not in self.ignore_letters]
        bag = np.zeros(len(self.words))
        for i, word in enumerate(self.words):
            if word in word_list:
                bag[i] = 1
        return bag

    def get_label(self, one_hot):
        index = np.argmax(one_hot)
        return self.classes[index]

    def predict(self, sentence):
        processed_data = self.preprocess(sentence)
        pred = self.model.predict(np.expand_dims(processed_data, axis=0))
        index = np.argmax(np.squeeze(pred))
        response = self.get_response(self.classes[index])

        return index, response


    def get_response(self, label):
        for intent in self.intents:
            if intent["tag"] != label:
                continue
            res_list = intent["responses"]
            index = np.random.randint(len(res_list))
            res = res_list[index]
            if res == "":
                return "Nothing"
            return res

    def run(self):
        print("Start chatting...")
        print("Type 'bye' to exit...")
        print()
        while True:
            sentence = input("You: ")
            if sentence == "bye":
                break

            processed_data = self.preprocess(sentence)
            pred = self.model.predict(np.expand_dims(processed_data, axis=0))
            label = self.get_label(pred[0])
            res = self.get_response(label)

            print("Chatbot:", res)
