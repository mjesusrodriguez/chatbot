from flask import Flask, render_template, request, url_for, flash, redirect, jsonify
from pymongo import MongoClient
import os
import openai
import json
import spacy
import random
import nltk
from nltk.corpus import wordnet
import numpy as np
import requests
import re
import en_core_web_sm

#nltk.download('wordnet')
#nltk.download('omw-1.4')

#from intentrec import intentRec

app = Flask(__name__)

client = MongoClient('localhost', 27017)
db = client.services
restaurant_sv = db.restaurant

# Set up the OpenAI API client
openai.api_key = "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"

# Set up the model and prompt
model_engine = "gpt-3.5-turbo-instruct"

#Variable global para guardar el servicio
service_id = ""
#variable global para guardar el intent
intent = ""

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

def slotFillingChatGPT(input, service, intent):
    f = open("services/"+service+".json", "r")
    json_wsl = " ".join(f.read().split())

    query = input
    #prompt = "Forget the information provided in our previous interactions. Using the following API specification " + json_wsl + " give me only a valid JSON format list with the value of the name of the parameters that are given and their value if they are not null (in a entry called 'filled') and the parameters that are not given (in a entry called 'empty') in the endpoint called /"+intent+"' with the following query: " + query
    #prompt = "Forget the information provided in our previous interactions. Consider the following API specification which contains an endpoint called /"+intent+"' with a list of parameters and their description:  " + json_wsl + ". Consider that you are a slot-filling chatbot that must obtain the value for those parameters, provided the following user's prompt: "+input+". Generate a valid JSON format with two entries: \"filled\" and \"empty\".  The entry \"empty\" contains the name and value of the rest of parameters, for all of them the value is \"null\"."
    prompt = "Forget the information provided in our previous interactions and pay attention only to the prompt: \""+input+"\". Consider the following API specification which contains an endpoint called /"+intent+"' with a list of parameters and their description:  " + json_wsl + ". Consider that you are a slot-filling chatbot that must obtain the value for those parameters, provided the following user's prompt: "+input+". Generate a valid JSON format with two entries: \"filled\" and \"empty\".  The entry \"empty\" contains the name and value of the rest of parameters that are \"null\" on \"filled\" entry, for all of them the value is \"null\"."
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
    intentServices = selectServiceByIntent(intent)
    #Busco en los servicios si hay alguno con esos tags, sino, los cojo todos.
    tagServices = filterServicesByTag(intentServices, tags)
    #Cojo uno aleatoriamente:
    #service_id = "63dd983b079fb348fceec1a8"
    print(tagServices)
    service_id = random.choice(tagServices)
    print(service_id)

    #consulto en los servicios que tengo que campos se han rellenado ya y cuales faltan y devuelvo las preguntas.
    slotFillingResponse = slotFillingChatGPT(userInput, service_id, intent)
    # Proceso JSON string
    print(slotFillingResponse)

    #cojo los parámetros vacíos procesando el string (menos errores de formato)
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

    #Lo paso a JSON
    emptyParams = json.loads(empty_string)
    print("EMPTY PARAMS")
    print(emptyParams)

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
    filledParams = filteredFilledParams
    print("FILLED PARAMS")
    print(filledParams)


    #hago una llamada a la función que dado un intent y un id me da las preguntas.
    intent_info = get_intent(service_id, intent)

    #Cuento la cantidad de parametros que hay en el json
    intent_info_json = intent_info[0].json
    slots = intent_info_json["intent"]["slots"]
    print("QUESTIONS FOR EACH SLOT")
    print(slots)

    json_slots = json.dumps(emptyParams)
    parsed_items = json.loads(json_slots)
    print("PARSED ITEMS")
    print(parsed_items)

    #Guardo las preguntas de los parámetros que hacen falta.
    questions = {}
    for empty in parsed_items:
        improved_question = improveQuestionchatGPT(slots[empty])
        questions[empty] = improved_question[2:]

    #return questions
    print("QUESTIONS")
    print(questions)

    return jsonify({'questions': questions},{'filled': filledParams},{'service_id': service_id}, {'intent': intent}), 202

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
        f = open('./services/' + data_from_client["service"] + '.json')

        # returns JSON object as a dictionary
        data = json.load(f)
        service_url = data['servers'][0]['url']
        intent = data_from_client["intent"]
        filleddata = data_from_client["filledSlots"]
        #Añado el email del usuario
        filleddata["email"] = data_from_client["email"]

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
    # opens JSON file
    f = open('./services/'+service_id+'.json')

    # returns JSON object as a dictionary
    data = json.load(f)

    # Itero el JSON y saco los intents que tiene definido el servicio
    for i in data['paths']:
        #Quito el primer caracter del endpoint que es un /
        intent_name_without_char = i.lstrip('/')
        #Selecciono el intent que coincide con el que se le envía en la ruta
        if (intent_name_without_char == intent_name):
            questions_resp = getQuestions(data['paths'][i]['get']['parameters'])
            intentInfo = {
                'name': intent_name_without_char,
                'description': data['paths'][i]['get']['description'],
                'slots': questions_resp
            }
        else:
            return jsonify({'msg': 'Intent not found'}), 404

    # Closing file
    f.close()
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
    """
    from flask_cors import CORS
    import ssl

    context = ssl.SSLContext()
    context.load_cert_chain("/home/mariajesus/certificados/conversational_ugr_es.pem",
                            "/home/mariajesus/certificados/conversational_ugr_es.key")
    CORS(app)
    app.run(host='0.0.0.0', port=5050, ssl_context=context, debug=False)
    """

    app.run()

