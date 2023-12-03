from bson import ObjectId, json_util
from flask import Flask, render_template, request, url_for, flash, redirect, jsonify
from pymongo import MongoClient
import os
import openai
import json
import spacy
from dateutil import parser
import random
import nltk
from nltk.corpus import wordnet
import numpy as np
import requests
import re

#nltk.download('wordnet')
#nltk.download('omw-1.4')

from intentrec import intentRec

app = Flask(__name__)

client = MongoClient('localhost', 27017)
db = client.services
restaurant_sv = db.restaurant

# Set up the OpenAI API client
openai.api_key = "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"

# Set up the model and prompt
model_engine = "gpt-3.5-turbo-instruct"

# Load the spaCy English model
nlp = spacy.load("en_core_web_sm")

#Variable global para guardar el servicio
service_id = ""
#variable global para guardar el intent
intent = ""

def transform_datetime (input_date, input_time):
    date_str = ""
    time_str = ""
    doc_date = nlp(input_date)
    for entity in doc_date.ents:
        if entity.label_ == "DATE":
            date_str = entity.text
            print("fecha")
            print(date_str)
            #date_obj = parser.parse(entity.text)

    doc_time = nlp(input_time)
    for entity in doc_time.ents:
        if entity.label_ == "TIME":
            time_str = entity.text
            print("hora")
            print(time_str)
            #date_obj = parser.parse(entity.text)
    datetime_str = f"{date_str} {time_str}"
    print(datetime_str)
    timestamp = parser.parse(datetime_str).timestamp()
    return timestamp

def remove_invisible_characters(input_string):
    # Define a regular expression pattern to match non-printable characters
    non_printable_pattern = re.compile(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]')

    # Use re.sub to replace matched characters with an empty string
    cleaned_string = re.sub(non_printable_pattern, '', input_string)

    return cleaned_string

def intentRecWithChatGPT(input):
    prompt="for the following input \""+input+"\" give me a JSON with only an intent of the user beetween those: BookRestaurant, PlayMusic, AddToPlayList, RateBook, SearchScreeningEvent, GetWeather, SearchCreativeWork"
    completion = openai.Completion.create(
        engine=model_engine,
        prompt=prompt,
        temperature=0.3,
        max_tokens=64,
        top_p=1,
        frequency_penalty=0.5,
        presence_penalty=0)
    response = completion.choices[0].text
    print(response)

    #Proceso JSON
    data = json.loads(response)
    intent = data["intent"]
    return intent

def selectServiceByIntent(intent):
    services = []
    # me meto en la carpeta servicios
    path_to_services = "./services/"
    # meto todos los ficheros
    for file in os.listdir(path_to_services):
        file_path = path_to_services + file

        f = open(file_path)
        data = json.load(f)
        for i in data['paths']:
            # Quito el primer caracter del endpoint que es un /
            intent_name_without_char = i.lstrip('/')
            # Selecciono el intent que coincide con el que se le envía en la ruta
            if (intent_name_without_char == intent):
                services.append(os.path.splitext(file)[0])
    return services

def selectServiceByIntentMongo(intent):
    services = []
    # Establishing connection with mongodb on localhost
    client = MongoClient('127.0.0.1', 27017)

    # Access database object
    db = client['services']

    # Access collection object
    collection = db['restaurant']

    # Retrieve all documents from the collection
    all_documents = collection.find()

    # Print or process each document as needed
    for document in all_documents:
        for i in document['paths']:
            # Quito el primer caracter del endpoint que es un /
            intent_name_without_char = i.lstrip('/')
            # Selecciono el intent que coincide con el que se le envía en la ruta
            if (intent_name_without_char == intent):
                services.append(document["_id"])
    return services

def filterServicesByTag(intentServices, userTags):
    tagServices = []

    for service_id in intentServices:
        # opens JSON file
        f = open('./services/' + service_id + '.json')

        # returns JSON object as a dictionary
        data = json.load(f)

        # Itero el JSON y saco los intents que tiene definido el servicio
        for i in data['tags']:
            tags = i["name"]

            #divido en tokens
            tagList = [substring.strip() for substring in tags.split(',')]

            for tag in userTags:
                if tag in tagList:
                    #si está el tag del usuario en los tags de los servicios guardo los servicios
                    tagServices.append(service_id)

    if len(tagServices) == 0:
        print("SERVICES ESTÁ VACÍO")
        # devuelvo entonces el mismo vector de servicios que tiene el intent porque no hay tags
        tagServices = intentServices

    return tagServices

def filterServicesByTagMongo(intentServices, userTags):
    tagServices = []
    # Establishing connection with mongodb on localhost
    client = MongoClient('127.0.0.1', 27017)

    # Access database object
    db = client['services']

    # Access collection object
    collection = db['restaurant']

    for service_id in intentServices:
        document = collection.find_one({"_id": ObjectId(service_id)})
        print("document tags")
        print(document["tags"])
        if "tags" in document:
            tags = document["tags"][0]["name"]

            # divido en tokens
            tagList = [substring.strip() for substring in tags.split(',')]

            for tag in userTags:
                if tag in tagList:
                    # si está el tag del usuario en los tags de los servicios guardo los servicios
                    tagServices.append(document["_id"])

    if len(tagServices) == 0:
        # devuelvo entonces el mismo vector de servicios que tiene el intent porque no hay tags
        tagServices = intentServices

    return tagServices

def get_synonyms(word):
    synonyms = []
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.append(lemma.name())
    return synonyms

def getTags(input):
    tags = []
    synonyms = []
    synonyms_ = []

    #utilizo spacy para el procesamiento de lenguaje natural
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(input)

    for token in doc:
        #cogeré los adjetivos
        if (token.pos_ == 'ADJ') or (token.pos_ == 'NOUN'):
            tags.append(token.text)

    for tag in tags:
        synonyms_ = []
        synonyms_ = get_synonyms(tag)
        for synonym in synonyms_:
            synonyms.append(synonym)

    return synonyms

def improveQuestionchatGPT(question):
    # manejar el prompt para que devuelva un json con parámetros faltantes.
    prompt = "Give me only one alternative question for this one in the scope of restaurant booking: " + question

    # Generate a response
    completion = openai.Completion.create(
        engine=model_engine,
        prompt=prompt,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.8,
    )

    response = completion.choices[0].text
    return response

def emptyParamsChatGPT(input, service, intent):
    f = open("services/"+service+".json", "r")
    json_wsl = " ".join(f.read().split())

    query = input
    prompt = "using the following API specification " + json_wsl + " give me only a JSON format list with the value of the name of the parameters that are not given in the endpoint called /"+intent+"' with the following query: " + query
    print(prompt)
    # Generate a response
    completion = openai.Completion.create(
        engine=model_engine,
        prompt=prompt,
        n=0,
        temperature=0,
    )
    response = completion.choices[0].text
    print("RESPUESTA")
    print(response)
    return response

def emptyParamsChatGPTMongo(input, service, intent):
    # Establishing connection with mongodb on localhost
    client = MongoClient('127.0.0.1', 27017)

    # Access database object
    db = client['services']

    # Access collection object
    collection = db['restaurant']

    objInstance = ObjectId(service)

    service = collection.find_one({"_id": objInstance})

    service_string = json.dumps(service, indent=4)
    print(service_string)

    query = input
    prompt = "using the following API specification " + service_string + " give me only a JSON format list with the value of the name of the parameters that are not given in the endpoint called /"+intent+"' with the following query: " + query
    print(prompt)
    # Generate a response
    completion = openai.Completion.create(
        engine=model_engine,
        prompt=prompt,
        n=0,
        temperature=0,
    )
    response = completion.choices[0].text
    print("RESPUESTA")
    print(response)
    return response

def slotFillingChatGPTMongo(input, service, intent):
    # Establishing connection with mongodb on localhost
    client = MongoClient('127.0.0.1', 27017)

    # Access database object
    db = client['services']

    # Access collection object
    collection = db['restaurant']

    objInstance = ObjectId(service)

    service = collection.find_one({"_id": objInstance})

    service_string = json_util.dumps(service)
    service_string_no_whitespace = service_string.replace(" ", "").replace("\n", "").replace("\t", "")
    print(service_string_no_whitespace)

    query = input
    #prompt = "Forget the information provided in our previous interactions. Using the following API specification " + json_wsl + " give me only a valid JSON format list with the value of the name of the parameters that are given and their value if they are not null (in a entry called 'filled') and the parameters that are not given (in a entry called 'empty') in the endpoint called /"+intent+"' with the following query: " + query
    #prompt = "Forget the information provided in our previous interactions. Consider the following API specification which contains an endpoint called /"+intent+"' with a list of parameters and their description:  " + json_wsl + ". Consider that you are a slot-filling chatbot that must obtain the value for those parameters, provided the following user's prompt: "+input+". Generate a valid JSON format with two entries: \"filled\" and \"empty\".  The entry \"empty\" contains the name and value of the rest of parameters, for all of them the value is \"null\"."
    prompt = "Forget the information provided in our previous interactions and pay attention only to the prompt: \""+input+"\". Consider the following API specification which contains an endpoint called /"+intent+"' with a list of parameters and their description:  " + service_string_no_whitespace + ". Consider that you are a slot-filling chatbot that must obtain the value for those parameters, provided the following user's prompt: "+input+". Generate a valid JSON format with two entries: \"filled\" and \"empty\".  The entry \"empty\" contains the name and value of the parameters that are null on the \"filled\" entry, for all of them the value is null."
    print(prompt)
    # Generate a response
    completion = openai.Completion.create(
        engine=model_engine,
        prompt=prompt,
        temperature=0,
        max_tokens=1024,
        n=1
    )
    response = completion.choices[0].text
    print("RESPUESTA CHATGPT")
    print(response)
    return response

def slotFillingChatGPT(input, service, intent):
    f = open("services/"+service+".json", "r")
    json_wsl = " ".join(f.read().split())

    query = input
    #prompt = "Forget the information provided in our previous interactions. Using the following API specification " + json_wsl + " give me only a valid JSON format list with the value of the name of the parameters that are given and their value if they are not null (in a entry called 'filled') and the parameters that are not given (in a entry called 'empty') in the endpoint called /"+intent+"' with the following query: " + query
    #prompt = "Forget the information provided in our previous interactions. Consider the following API specification which contains an endpoint called /"+intent+"' with a list of parameters and their description:  " + json_wsl + ". Consider that you are a slot-filling chatbot that must obtain the value for those parameters, provided the following user's prompt: "+input+". Generate a valid JSON format with two entries: \"filled\" and \"empty\".  The entry \"empty\" contains the name and value of the rest of parameters, for all of them the value is \"null\"."
    prompt = "Forget the information provided in our previous interactions and pay attention only to the prompt: \""+input+"\". Consider the following API specification which contains an endpoint called /"+intent+"' with a list of parameters and their description:  " + json_wsl + ". Consider that you are a slot-filling chatbot that must obtain the value for those parameters, provided the following user's prompt: "+input+". Generate a valid JSON format with two entries: \"filled\" and \"empty\".  The entry \"empty\" contains the name and value of the parameters that are \"null\" on the \"filled\" entry, for all of them the value is \"null\"."
    print(prompt)
    # Generate a response
    completion = openai.Completion.create(
        engine=model_engine,
        prompt=prompt,
        temperature=0,
        max_tokens=1024,
        n=1
    )
    response = completion.choices[0].text
    print("RESPUESTA CHATGPT")
    print(response)
    return response

#Sin usar
def servicecall(id, intent):
    #Cojo el servicio
    service_json = restaurant_sv.find_one({'_id': id})

    #De los paths busco el intent
    service = json.loads(service_json)

    paths = service["paths"]
    for path in paths:
        if(path == intent):
            #cojo las preguntas
            intent = path
            questions = intent["get"]["parameters"]
            return questions

@app.route('/')
def index():
    return render_template('ltv2.html')

@app.route('/chatbot', methods=['GET'])
def create():
    global json_object
    userInput = request.args.get('input')

    #intent = "bookrestaurant"
    user_intent = intentRecWithChatGPT(userInput)
    intent = user_intent.lower()
    #intentPrediction = intentRec(userInput)
    #intent = intentPrediction[0]

    #Saco los tags del input del usuario
    tags = getTags(userInput)

    #Busco todos los servicios que tengan ese intent
    intentServices = selectServiceByIntentMongo(intent)
    #Busco en los servicios si hay alguno con esos tags, sino, los cojo todos.
    tagServices = filterServicesByTagMongo(intentServices, tags)
    #Cojo uno aleatoriamente:
    print(tagServices)
    service_id = random.choice(tagServices)
    print(service_id)

    #consulto en los servicios que tengo que campos se han rellenado ya y cuales faltan y devuelvo las preguntas.
    slotFillingResponse = slotFillingChatGPTMongo(userInput, service_id, intent)

    #cojo los parámetros vacíos procesando el string (menos errores de formato)
    #Esto no lo devuelve bien ChatGPT, no lo estoy usando
    empty_string = ""
    start_index = slotFillingResponse.find('"empty":')
    if start_index != -1:
        start_index = slotFillingResponse.find('{', start_index)
        end_index = slotFillingResponse.find('}', start_index)
        empty_string = slotFillingResponse[start_index:end_index + 1]

        # Now you have the "empty" object as a string
        print(empty_string)
    else:
        print('"empty" not found in the string')

    #Cojo los parámetros rellenos procesando el string (menos errores de formato)
    filled_string = ""
    start_index = slotFillingResponse.find('"filled":')
    if start_index != -1:
        start_index = slotFillingResponse.find('{', start_index)
        end_index = slotFillingResponse.find('}', start_index)
        filled_string = slotFillingResponse[start_index:end_index + 1]

        # Now you have the "filled" object as a string
        print(filled_string)
    else:
        print('"filled" not found in the string')

    # Lo paso a JSON
    filledParams = json.loads(filled_string)
    #Si alguno tiene el valor none, lo quito.
    filteredFilledParams = {key: value for key, value in filledParams.items() if value is not None}
    filteredNotFilledParams = {key: value for key, value in filledParams.items() if value is None}
    filledParams = filteredFilledParams
    notFilledParams = filteredNotFilledParams
    print("FILLED PARAMS")
    print(filledParams)
    print("NOT FILLED PARAMS")
    print(notFilledParams)


    #hago una llamada a la función que dado un intent y un id me da las preguntas.
    intent_info = get_intent(service_id, intent)

    #Cuento la cantidad de parametros que hay en el json
    intent_info_json = intent_info[0].json
    slots = intent_info_json["intent"]["slots"]
    #print("QUESTIONS FOR EACH SLOT")
    #print(slots)

    #json_slots = json.dumps(emptyParams)
    json_slots = json.dumps(notFilledParams)
    parsed_items = json.loads(json_slots)
    print("PARSED ITEMS")
    print(parsed_items)

    #Guardo las preguntas de los parámetros que hacen falta.
    questions = {}
    for empty in parsed_items:
        questions[empty] = improveQuestionchatGPT(slots[empty])

    #return questions
    print("QUESTIONS")
    print(questions)

    return jsonify({'questions': questions},{'filled': filledParams},{'service_id': str(service_id)}, {'intent': intent}), 202

@app.route('/serviceinfo/data', methods=['POST'])
def data():
    try:
        # Get the JSON data from the request
        data_from_client = request.get_json()

        #Hago lo que sea con la información del cliente
        print("PARÁMETROS DESDE CLIENTE RECOGIDOS")
        print(data_from_client)

        #Busco el server del servicio elegido.
        # Abro el servicio para hacer la llamada
        # Establishing connection with mongodb on localhost
        client = MongoClient('127.0.0.1', 27017)

        # Access database object
        db = client['services']

        # Access collection object
        collection = db['restaurant']

        service_id = ObjectId(data_from_client["service"])

        service = collection.find_one({"_id": service_id})

        service_url = service['servers'][0]['url']
        intent = data_from_client["intent"]
        filleddata = data_from_client["filledSlots"]
        #Añado el email del usuario
        filleddata["email"] = data_from_client["email"]

        if "date" in filleddata and "time" in filleddata:
            print("entro en flujo de control")
            nuevafecha = transform_datetime(filleddata["date"], filleddata["time"])
            print(nuevafecha)
            filleddata["datetime"] = nuevafecha
            print("DATOS PARA EL SERVICIO")
            print(filleddata)

        # Cojo la ruta del server del JSON
        route = service_url + "/" + intent
        print(route)

        # Send the POST request
        response = requests.post(route, json=filleddata)

        # Check the response
        if response.status_code == 200:
            print("POST request was successful.")
            print("Response content:", response.text)
        else:
            print("POST request failed with status code:", response.status_code)

        #return jsonify({"message": "Data updated successfully"})

    except Exception as e:
        return jsonify({"error": str(e)})


#Le paso el intent y el servicio a esta función que representará los ENDPOINT del servicio.
@app.route('/intentinfo/<service_id>/intent/<intent_name>')
def get_intent(service_id, intent_name):
    # Establishing connection with mongodb on localhost
    client = MongoClient('127.0.0.1', 27017)

    # Access database object
    db = client['services']

    # Access collection object
    collection = db['restaurant']

    objInstance = ObjectId(service_id)

    service = collection.find_one({"_id": objInstance})

    # Itero el JSON y saco los intents que tiene definido el servicio
    for i in service['paths']:
        #Quito el primer caracter del endpoint que es un /
        intent_name_without_char = i.lstrip('/')
        #Selecciono el intent que coincide con el que se le envía en la ruta
        if (intent_name_without_char == intent_name):
            questions_resp = getQuestions(service['paths'][i]['get']['parameters'])
            intentInfo = {
                'name': intent_name_without_char,
                'description': service['paths'][i]['get']['description'],
                'slots': questions_resp
            }
        else:
            return jsonify({'msg': 'Intent not found'}), 404

    try:
        return jsonify({'intent': intentInfo}), 202
    except:
        return jsonify({'msg': 'Error finding intent'}), 500

#cojo las preguntas de un conjunto de parámetros que se le pasa de un intent
@app.route('/questions/<parameters>')
def getQuestions(parameters):
    slots = {}
    for i in parameters:
        slots[i['name']] = i['x-custom-question']
    try:
        return slots
    except Exception:
        return jsonify({'msg': 'Error finding questions'}), 500

if __name__ == '__main__':
    app.run()

