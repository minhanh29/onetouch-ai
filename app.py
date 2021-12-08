from flask import Flask, request, jsonify, render_template
from flask_cors import CORS, cross_origin
from model.chatbot import ChatBot


app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


@app.route('/')
@cross_origin()
def home():
    return render_template('index.html')


@app.route('/recognize', methods=['POST'])
@cross_origin()
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
    app.run(debug=True, use_reloader=True)
