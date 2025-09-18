# from fastapi import FastAPI
# from pydantic import BaseModel
# from bot_logic import run_conversation, messages

# app = FastAPI()

# class Query(BaseModel):
#     question: str
    
# @app.post("/ask")
# def ask_bot(query: Query):
#     response = run_conversation(messages, query.question)
#     answer = response.choices[0].message.content
#     return {"question" : query.question, "answer" : answer}


from flask import Flask, render_template, request, jsonify

from bot_logic import run_conversation, messages

from flask_cors import CORS

app = Flask(__name__)

# Flask
CORS(app, resources={r"/*": {"origins": "*"}})

# === Replace this function with your actual bot logic ===
# def bot_logic(user_input, context=None):
#     # Example dummy logic
#     response = f"You said: {user_input}"
#     return {"response": response, "context": context}

# @app.route('/chat', methods=['POST'])
# def chat():
#     data = request.get_json()
#     user_input = data.get('message')
#     context = data.get('context', None)

#     if not user_input:
#         return jsonify({'error': 'No message provided'}), 400

#     bot_reply = bot_logic(user_input, context)
#     return jsonify(bot_reply)

# # === API route for interacting with the bot ===
@app.route("/get")
def get_bot_response():    
    userText = request.args.get('msg')  
    returned = run_conversation(messages, userText)
    if type(returned) == str:
        return returned
    response = returned.choices[0].message.content
    #return str(bot.get_response(userText)) 
    return response

# === Health check or root route ===
@app.route('/')
def index():
    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)
