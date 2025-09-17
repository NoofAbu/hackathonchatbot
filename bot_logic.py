import os
import json
import requests
import datetime
import pandas as pd
import csv
from PIL import Image
import urllib.request
# import speech_recognition as sr
from gtts import gTTS
# from playsound import playsound
import faiss
import numpy as np

from openai import AzureOpenAI
from langchain.schema import Document

# --------------------
# Azure OpenAI Setup
# --------------------
# endpoint = "https://hiaeastus2.openai.azure.com/"
# endpoint = "https://noofm-mfmch2ss-eastus2.cognitiveservices.azure.com/openai/v1/"
endpoint = "https://hackathon-azure-ai-foundry25.openai.azure.com/"
model_name = "gpt-4"
deployment = "gpt-4"  # your chat deployment name
embedding_deployment = "text-embedding-3-small"  # your embedding deployment name
# subscription_key = "704134ef47b84defbc2c0a5213928da6"
# subscription_key = "CDBPJd6TOeDe570cJkK5ApQnO2Kqb0brEKuCtBwf9tUNVGIrFvqGJQQJ99BIACHYHv6XJ3w3AAAAACOGWpg8"
subscription_key = "EHnTkaGx14Eo8YOnl1KRVOISrfTxZC4NiMnsp4fe6TwZ0MMFsgFPJQQJ99BIAC5RqLJXJ3w3AAAAACOG7Kxn"
api_version = "2024-12-01-preview"

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)

# recognizer = sr.Recognizer()
messages = []

# --------------------
# Speech Helpers
# --------------------
# def text_to_speech(response_text):
#     tts = gTTS(text=response_text, lang='en')
#     tts.save("response.mp3")
#     # playsound("response.mp3")

# def listen_for_question():
#     with sr.Microphone() as source:
#         # playVideo("chatbot_hello.mp4")
#         print("Listening for a question...")
#         # audio = recognizer.listen(source)
#     try:
#         # question = recognizer.recognize_google(audio)
#         print("You:", question)
#         return question
#     except sr.UnknownValueError:
#         print("Sorry, I couldn't understand your question.")
#         # playVideo("chatbot_sorry.mp4")
#         return None

# --------------------
# CSV Data Prep
# --------------------
def get_concourse(visioIDs):
    locations = []
    for visioId in visioIDs:
        if visioId.startswith("B01-UL001-IDA") or visioId.startswith("B01-UL002-IDA"):
            locations.append("A concourse")
        if visioId.startswith("B01-UL001-IDB") or visioId.startswith("B01-UL002-IDB"):
            locations.append("B concourse")
        if visioId.startswith("B01-UL001-IDC") or visioId.startswith("B01-UL002-IDC"):
            locations.append("C concourse")
        if visioId.startswith("B01-UL001-IDL") or visioId.startswith("B01-UL002-IDL"):
            locations.append("Landside")
        if visioId.startswith("B01-UL000"):
            locations.append("Ground Floor")
        if visioId.startswith("B01-UL001"):
            locations.append("1st Floor")
        if visioId.startswith("B01-UL002"):
            locations.append("2nd Floor")
    return locations

def get_content_response():
    # api_url = "https://dohahamadairport.com/api/content.json?androidversion=3.3"
    api_url = "https://dohahamadairport.com/api/v2/content"
    response = requests.get(api_url)
    data = json.loads(response.text)
    df = pd.json_normalize(data)
    df.fillna('', inplace=True)
    df.rename(columns={
        "mcn_map_location": "visioglobe"
    }, inplace=True)
    df = df[['mcn_nid','mcn_ntype', 'mcn_content', 'visioglobe']]
    df['mcn_title_en'] = df['mcn_content'].apply(lambda x: x[0]["mcn_title"])
    df['description_en'] = df['mcn_content'].apply(lambda x: x[0]["mcn_body"])
    df['meta_tags'] = df['mcn_content'].apply(lambda x: x[0]["mcn_meta_tags"])
    df['location'] = df['visioglobe'].apply(get_concourse)
    df = df.query("mcn_ntype in ['shop','dine','relax','art','facilities','lounge','page','gates_and_belts']")
    df.to_csv("content_data.csv", index=False, encoding='utf-8')

def read_csv_into_vector_document(file, text_cols):
    with open(file, newline='') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        text_data = []
        for row in csv_reader:
            text = ' '.join([str(row[col]) for col in text_cols])
            text_data.append(text)
        return [Document(page_content=text) for text in text_data]

get_content_response()
data = read_csv_into_vector_document("content_data.csv", [ 'mcn_nid','mcn_ntype','mcn_title_en','description_en','meta_tags','visioglobe','location'])

# --------------------
# Embeddings + FAISS
# --------------------
def embed_texts(texts):
    response = client.embeddings.create(
        model=embedding_deployment,
        input=texts
    )
    return [np.array(item.embedding, dtype="float32") for item in response.data]

embeddings = embed_texts([doc.page_content for doc in data])
dim = len(embeddings[0])
index = faiss.IndexFlatL2(dim)
index.add(np.array(embeddings))

def search_docs(query, top_k=3):
    q_emb = embed_texts([query])[0]
    D, I = index.search(np.array([q_emb]), top_k)
    return [data[i].page_content for i in I[0]]

# --------------------
# Functions
# --------------------
# def get_flight_information(arguments):
#     try:
#         args = json.loads(arguments)
#         if 'query' in args:
#             query = args['query'].replace(" ", "")
#             payload = {"globalSearch": query, "type": "departures", "limit": "1", "startTime": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), "endTime": datetime.datetime.now().strftime('%Y-%m-%d 23:59')}
#         elif 'gate_query' in args:
#             payload = {"gateNo": args['gate_query'], "type": "departures", "limit": "1", "startTime": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), "endTime": datetime.datetime.now().strftime('%Y-%m-%d 23:59')}
#         else:
#             return "{}"

#         r = requests.post("https://dohahamadairport.com/webservices/fids", headers={"Content-Type": "application/json"}, json=payload)
#         flights = r.json()['flights']
#         select_list = ['flightNumber','airlineCode','destinationCode','originCode','scheduledTime']
#         getdata = [{x: each[x] for x in select_list if x in each} for each in flights]
#         return json.dumps(getdata)
#     except:
#         return "Could not retrieve flights"

def get_content_details(arguments):
    args = json.loads(arguments)
    docs = search_docs(args['content_query'], top_k=3)
    return json.dumps(docs)

# def get_location_details(arguments):
#     visio_id = json.loads(arguments)['visio_query']
#     qrcode_data = f'https://dohahamadairport.com/wayfinding/routecode/?dst={visio_id}&src=B01-UL001-IDA0379'
#     urllib.request.urlretrieve(qrcode_data, "MyQRCode1.png")
#     my_img = Image.open("MyQRCode1.png")
#     my_img.show()
#     return "Directions QR generated"

# --------------------
# Conversation with Function Calling
# --------------------
    
def run_conversation(messages, question):
    messages.append({"role": "user", "content": question})
    response = client.chat.completions.create(
        model=deployment,
        messages=messages,
        functions=[
            # {
            #     "name": "get_flight_information",
            #     "description" : f"""You are a customer service agent at Hamad International Airport, Doha - Qatar. 
            #     The current date and time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.
            #     Get flight information based on user's query. This query can be a destination or a city or a flight number.
            #     It can also be a gate for which the user wants to know next flights departing from.
            #     Give a short reply with flight date, time and boarding gate information only. Do not give anything else as response.""",
            #     "parameters": {
            #         "type": "object",
            #         "properties": {
            #             "query": {
            #                 "type": "string",
            #                 "description": "This is the destination to which the user wants to travel to or the flight number.",
            #             },
            #             "gate_query": {
            #                 "type": "string",
            #                 "description": "This is the gate for which the user wants to know about upcoming flights",
            #             },
            #             "scheduled_time" : {
            #                 "type": "string",
            #                 "description": "This is the scheduled flight time in unix timestamp",
            #             }
            #         },
            #         # "required": ["scheduled_time"],
            #     },
            # },
            {
                "name": "get_content_details",
                "description" : f"""Current date and time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.
                Use this fucntion when the user query is not relacted to flights. It can be related to shops, restaurants, facilities, lounges, gate location etc inside the airport.
                Give a very short response to the user in less than 15 words. Do not specify location ids in the response.
                Do not provide any other information unless the user asks for it.
                """,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content_query": {
                            "type": "string",
                            "description": "This is keyword related to airport shop, dine, facilities etc to be searched. This cannot be a flight related keyword",
                        },
                        
                    },
                    "required": ["content_query"],
                },
            }
            # {
            #     "name": "get_location_details",
            #     "description" : f"""Current date and time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.
            #     Get directions only if the user asks for directions to a particular boarding gate.
            #     If the customer asks the way to the shop, restaurant, facility or boarding gate, do not give id back in response.
            #     Only ask if the information provided was helpful and if there is anything else you can help them with.
            #     """,
            #     "parameters": {
            #         "type": "object",
            #         "properties": {
            #             "visio_query": {
            #                 "type": "string",
            #                 "description": "This is the visioglobe id related to airport shop, dine, facilities that the user enquired about.\
            #                     If it is related to boarding gate take the gateLocationId from data.\
            #                     When user queries boarding gate location information use this. This value will be used to determine the directions.",
            #             },
                        
            #         },
            #         "required": ["visio_query"],
            #     },
            # }
        ],
        function_call="auto",
    )
    message = response.choices[0].message
    messages.append(message)

    if hasattr(message, "function_call") and message.function_call is not None:
        available_functions = {
        # "get_flight_information": get_flight_information,
        "get_content_details": get_content_details,
        # "get_location_details": get_location_details
    }
        function_name = message.function_call.name
        function_to_call = available_functions[function_name]
        function_arguments = message.function_call.arguments
        function_response = function_to_call(function_arguments)

        messages.append({"role": "system", "content": "Always answer in less than 30 words."})
        messages.append({"role": "function","name": function_name,"content": function_response})

        second_response = client.chat.completions.create(model=deployment, messages=messages)
        return second_response
        
    return response

# --------------------
# Main Loop
# --------------------
#uncomment to run file
# while True:
#     question = input("Say something: ")
#     if question:
#         returned = run_conversation(messages, question)
#         answer = returned.choices[0].message.content
#         print("Bot:", answer)