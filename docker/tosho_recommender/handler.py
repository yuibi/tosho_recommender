import numpy as np
import pandas as pd
import pickle
import gensim
import janome
from janome.tokenizer import Tokenizer as ja_tokenizer
import re
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from model import *
    
def load_obj(filename):
    with open(filename, 'rb') as handler:
        return pickle.load(handler)

# Load tokenizer
tokenizer = load_obj('model/tosho_recommender_word_tokenizer.pkl')

# Load Book2Vec (book-level representation)
b2v = gensim.models.keyedvectors.KeyedVectors.load_word2vec_format('model/book2vec', binary=False)

# Load book descriptions and titles
text_pd = pd.read_csv('data/tosho_processed_clean.csv.bz2', sep='\t', compression='bz2')

# Load tf.keras model
model = gru_model(
                     embedding_dim=300,
                     dropout_rate=0.209,
                     rnn_unit=194,
                     input_shape=(500,),
                     num_features=20000+1,
                     share_gru_weights_on_book=True,
                     use_attention_on_book=True,
                     use_attention_on_user=True,
                     use_batch_norm=False,
                     is_embedding_trainable=False,
                     final_activation='tanh',
                     final_dimension=392,
                     embedding_matrix=np.zeros((20001, 300)))

model.compile(loss='cosine_similarity',
                  optimizer=Adam(lr=0.0044))

x = np.ones((1, 500))
y = np.ones((1, 392))

model.train_on_batch([x, x, x, x], y)

# Load the state of the old model
model.load_weights('model/tosho_recommender_7')

# Clean raw text   
def clean_text(text):
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\s{1,}', '', text)
    
    text = re.sub(r'内容紹介', '', text)
    text = re.sub(r'出版社からのコメント', '', text)
    text = re.sub(r'商品の説明をすべて表示する', '', text)
    text = re.sub(r'内容（「MARC」データベースより）', '', text)
    text = re.sub(r'内容（「BOOK」データベースより）', '', text)

    non_japanese = re.compile(r"[^0-9\-ぁ-ヶ亜-黑ー]")
    text = re.sub(non_japanese, ' ', text)

    return text.strip()

# Tokenize Japanese text
j_tokenizer = ja_tokenizer()
def wakati_reading(text):
    tokens = j_tokenizer.tokenize(text.replace("'", "").lower())
    
    exclude_pos = [u'助動詞']
    
    #分かち書き
    tokens_w_space = ""
    for token in tokens:
        partOfSpeech = token.part_of_speech.split(',')[0]
        
        if partOfSpeech not in exclude_pos:
            tokens_w_space = tokens_w_space + " " + token.surface

    tokens_w_space = tokens_w_space.strip()
    tokens_w_space = re.sub(r'\s{2,}', ' ', tokens_w_space)
    
    return tokens_w_space

# stoi (string to integer) and padding
def preprocess_text(text:str):
    MAX_SEQUENCE_LENGTH = 500
    
    text = clean_text(text)
    text = wakati_reading(text)

    x = tokenizer.texts_to_sequences([text])
    x = pad_sequences(x, maxlen=MAX_SEQUENCE_LENGTH)
    
    return x

# Recommend 3 books
def recommend_book(book_list:list=None):
    book_vec = model.predict([preprocess_text(book_list[0]), preprocess_text(book_list[1]), preprocess_text(book_list[2]), preprocess_text(book_list[3])])
    book_w2v_result = b2v.similar_by_vector(book_vec[0], topn=3)
                  
    similar_book_id_1 = book_w2v_result[0][0]
    similar_book_title_1 = text_pd[text_pd.id == int(similar_book_id_1)]['title'].values[0]
    similar_book_desc_1 = text_pd[text_pd.id == int(similar_book_id_1)]['description_token'].values[0]
    similar_book_id_2 = book_w2v_result[1][0]
    similar_book_title_2 = text_pd[text_pd.id == int(similar_book_id_2)]['title'].values[0]
    similar_book_desc_2 = text_pd[text_pd.id == int(similar_book_id_2)]['description_token'].values[0]
    similar_book_id_3 = book_w2v_result[2][0]
    similar_book_title_3 = text_pd[text_pd.id == int(similar_book_id_3)]['title'].values[0]
    similar_book_desc_3 = text_pd[text_pd.id == int(similar_book_id_3)]['description_token'].values[0]

    return {"1st book": {"id": similar_book_id_1, "title": similar_book_title_1, "desc": similar_book_desc_1}, "2nd book": {"id": similar_book_id_2, "title": similar_book_title_2, "desc": similar_book_desc_2}, "3rd book": {"id": similar_book_id_3, "title": similar_book_title_3, "desc": similar_book_desc_3}}