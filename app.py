

import tensorflow as tf


from flask import Flask, jsonify, request
import numpy as np
from sklearn.externals import joblib
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.externals import joblib
from bs4 import BeautifulSoup
import re
from pickle import load
from flask import Flask, request, jsonify, render_template


from sklearn.feature_extraction.text import CountVectorizer
import os
os.chdir('/Users/subham/Desktop/ImageCaptioning/Flickr8k')


from image_captnng import *

# https://www.tutorialspoint.com/flask
import flask
app = Flask(__name__)




@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/index')
def index():
    return flask.render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    checkpoint_path = "./checkpoint_finally7/train/ckpt-4"
    train_captions = load(open('./captions.pkl', 'rb'))


    # Choose the top 5000 words from the vocabulary
    top_k = 5000
    train_seqs , tokenizer = tokenize_caption(top_k ,train_captions)

    #restoring the model

    ckpt = tf.train.Checkpoint(encoder=encoder,decoder=decoder,optimizer = optimizer)
    ckpt.restore(checkpoint_path)

    to_predict_list = request.form.to_dict()
    Image_path = to_predict_list['pic_url']

    def evaluate(image):
        attention_plot = np.zeros((max_length, attention_features_shape))

        hidden = decoder.reset_state(batch_size=1)

        temp_input = tf.expand_dims(load_image(image)[0], 0)
        img_tensor_val = image_features_extract_model(temp_input)
        img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

        features = encoder(img_tensor_val)

        dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
        result = []

        for i in range(max_length):
            predictions, hidden, attention_weights = decoder(dec_input, features, hidden)

            attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()

            predicted_id = tf.argmax(predictions[0]).numpy()
            result.append(tokenizer.index_word[predicted_id])

            if tokenizer.index_word[predicted_id] == '<end>':
                return result, attention_plot

            dec_input = tf.expand_dims([predicted_id], 0)

        attention_plot = attention_plot[:len(result), :]
        return result, attention_plot


    new_img =  Image_path
    #image_extension = Image_path[-3:]

    #image_path = tf.keras.utils.get_file('image'+image_extension,origin=Image_path)

    result, attention_plot = evaluate(new_img)
    for i in result:
        if i=="<unk>":
            result.remove(i)
        else:
            pass

    #print('I guess: ', ' '.join(result).rsplit(' ', 1)[0])
    captn =' '
    return render_template('index.html', prediction_text='Predicted Caption : {}'.format(captn.join(result).rsplit(' ', 1)[0]))




if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
