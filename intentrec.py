"""
#Documento para pasar el modelo pre-entrenado BERT para que me saque la predicción del intent

#Importo librerías necesarias para el clasificador
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from bert.loader import StockBertConfig, map_stock_config_to_params
from bert import BertModelLayer
from bert.loader import load_stock_weights
from sklearn.metrics import classification_report
#Import FullTokenizer from bert_tokenization
from bert.tokenization.bert_tokenization import FullTokenizer

def intentRec(input):
    inputFolder = 'input/intent-recognition/'

    train = pd.read_csv(inputFolder + "train.csv")
    valid = pd.read_csv(inputFolder + "valid.csv")
    test = pd.read_csv(inputFolder + "test.csv")

    train = pd.concat([train, valid], ignore_index = True)

    #print the unique intents
    intents = train.intent.unique()
    intent_number = train.intent.value_counts()

    sns.set()
    plt.figure(figsize = (12, 8))
    chart = sns.countplot(x = 'intent', data = train, palette='Set1')
    chart.set_xticklabels(chart.get_xticklabels(), rotation = 30, horizontalalignment='right', fontweight='light', fontsize='medium')
    chart.set_title('Intent Distribution', fontsize = 18)
    chart.set_xlabel('Intents', fontsize = 14)
    chart.set_ylabel('Counts', fontsize = 14)
    #plt.show()

    #aquí cargo el modelo BERT
    modelInputFolder = 'input/'
    bert_model_name="uncased_L-12_H-768_A-12"
    bert_ckpt_dir = os.path.join(modelInputFolder, bert_model_name)
    bert_ckpt_file = os.path.join(bert_ckpt_dir, "bert_model.ckpt")
    bert_config_file = os.path.join(bert_ckpt_dir, "bert_config.json")

    vocab_file = os.path.join(bert_ckpt_dir, "vocab.txt")

    tokenizer = FullTokenizer(vocab_file)
    #print(tokenizer)

    tokens = tokenizer.tokenize("Hello, How are you?")
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    #print(tokens)
    #print(token_ids)

    classes = train.intent.unique().tolist()
    #print(classes)

    max_seq_len = 192
    #pass dataframe as input in lambda function
    #sort values by length of text index
    #and return the new index of sorted_indexes
    sort_by_length_text = lambda input_df: input_df.reindex(
        input_df['text'].str.len().sort_values().index
    )
    #print(sort_by_length_text)

    #PREPROCESS DATA
    class IntentDataManager:

        def __init__(self, train, test, tokenizer: FullTokenizer, classes, max_seq_len):

            # declare tokenizer and classes as a class members
            self.tokenizer = tokenizer
            self.classes = classes
            self.max_seq_len = 0

            # sort train and test data by length of text
            train, test = map(sort_by_length_text, [train, test])

            # call preprocessData function
            (self.train_X, self.train_y), (self.test_X, self.test_y) = map(self.preprocessData, [train, test])

            self.max_seq_len = min(self.max_seq_len, max_seq_len)
            self.train_X, self.test_X = map(self.padSequences, [self.train_X, self.test_X])

            pass

        def preprocessData(self, df):

            x, y = [], []
            for idx, row in df.iterrows():
                text = row['text']
                label = row['intent']

                # convert text to tokens
                tokens = self.tokenizer.tokenize(text)
                tokens = ["[CLS]"] + tokens + ["[SEP]"]

                # convert tokens to ids
                token_ids = self.tokenizer.convert_tokens_to_ids(tokens)

                # append tokens_ids to x
                x.append(token_ids)
                # get maxmium sequence length
                self.max_seq_len = max(self.max_seq_len, len(token_ids))

                # get index of class label
                class_label_index = self.classes.index(label)

                # append index of class label to y
                y.append(class_label_index)
                pass

            print(x)
            arrX = np.array(x)
            arrY = np.array(y)

            return arrX, arrY
            pass

        def padSequences(self, arr):

            # print("arr", arr)
            newArr = []
            for item in arr:
                # print("item", item)
                # calculate the shortfall of sequence length
                shortfall = self.max_seq_len - len(item)

                # add zero to shortfall
                # zerosToAdd = np.zeros(shortfall, dtype = np.int32)
                # newItem = np.append(item, zerosToAdd)
                item = item + [0] * (shortfall)
                # print(newItem)

                newArr.append(np.array(item))
                pass

            return np.array(newArr)
            pass

        pass


    # class IntentDataManager:
    data = IntentDataManager(train, test, tokenizer, classes, max_seq_len)
    # print(data)

    #print(data.train_X.shape)
    #print(data.train_X[0])
    #print(data.test_X[0])


    # create customModel
    def customModel(max_seq_len, bert_config_file, bert_ckpt_file):
        # create input layer
        input_layer = keras.layers.Input(
            shape=(max_seq_len,),
            dtype='int32',
            name="input_layer")
        # read config file with special reader tf.io.gfile.GFile
        with tf.io.gfile.GFile(bert_config_file, "r") as reader:
            # read data as json string
            customConfig = StockBertConfig.from_json_string(reader.read())

            # load all params for our model
            # If params not in customConfig, defauls value is used
            bert_params = map_stock_config_to_params(customConfig)

            # print(f"\nbert_params.adapter_size = {bert_params.adapter_size}")
            bert_params.adapter_size = None

            # create bert layer
            bert_layer = BertModelLayer.from_params(bert_params, name="bert_layer")

            pass

        # process input through bert_layer
        bert_output = bert_layer(input_layer)

        # add hidden layer1
        hidden_output1 = keras.layers.Lambda(lambda seq: seq[:, 0, :])(bert_output)

        # dropout layer 1
        dropout_1 = keras.layers.Dropout(0.5)(hidden_output1)
        # add hidden layer2
        hidden_output2 = keras.layers.Dense(units=768, activation="tanh")(dropout_1)

        # dropout layer 2
        dropout_2 = keras.layers.Dropout(0.5)(hidden_output2)

        final_output = keras.layers.Dense(units=len(classes), activation="softmax")(dropout_2)
        # create model with all layers
        model = keras.Model(inputs=input_layer, outputs=final_output)
        model.build(input_shape=(None, max_seq_len))

        load_stock_weights(bert_layer, bert_ckpt_file)

        return model
        pass
    pass

    model = customModel(data.max_seq_len, bert_config_file, bert_ckpt_file)

    model.compile(
        optimizer=keras.optimizers.Adam(1e-5),

        # Loss function to minimize
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),

        # List of metrics to monitor
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")]
    )

    x = data.train_X
    y = data.train_y
    history = model.fit(x, y, validation_split = 0.1, batch_size = 16, shuffle = True, epochs = 5)

    train_loss, train_accuracy = model.evaluate(data.train_X, data.train_y)
    test_loss, test_accuracy = model.evaluate(data.test_X, data.test_y, batch_size = 16)
    print("train_loss, train_accuracy:", train_accuracy)
    print("test_loss, test_accuracy:", test_accuracy)

    #predict test data
    y_pred = model.predict(data.test_X).argmax(axis = -1)
    print(y_pred.shape)
    print(y_pred[:10])

    for label in y_pred[:10]:
        print(classes[label])
        pass

    print(classification_report(data.test_y, y_pred, target_names = classes))

    sentences = [
        "I want a to book a restaurant in south zone",
        "I want to look for spanish food",
        "I want to eat mexican food"
    ]
    #tokenize sentences
    tokens = map(tokenizer.tokenize, input)
    #add [CLS] and [SEP] Tokens
    tokens = map(lambda token: ["[CLS]"] + token + ["[SEP]"], tokens)
    #convert each tokens to idsp
    token_ids = list(map(tokenizer.convert_tokens_to_ids, tokens))
    #add padding
    token_ids = map(lambda tids: tids + [0] * (data.max_seq_len-len(tids)), token_ids)
    token_ids = np.array(list(token_ids))
    #predict
    predictions = model.predict(token_ids).argmax(axis = -1)
    print("PREDICTION: " + predictions)
    return [predictions, classes]
    #for text, label in zip(input, predictions):
        #print("Text:", text, "\nIntent:", classes[label])
        #print()
"""