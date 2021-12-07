from flask import Flask, request, jsonify, render_template
from model.chatbot import ChatBot


app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/recognize', methods=['POST'])
def recognize():
    data = request.get_json(force=True)
    sentence = data['text']

    index, response = ChatBot.getInstance().predict(sentence)

    output = {
        'id': int(index),
        'response': response
    }

    return jsonify(output)


if __name__ == '__main__':
    ChatBot.getInstance().predict("xin chào")
    app.run(debug=False, use_reloader=True)
