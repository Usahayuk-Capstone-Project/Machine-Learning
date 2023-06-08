#!/usr/bin/env python
# coding: utf-8

# In[1]:


from uuid import uuid4
from fastapi import FastAPI, Form
from fastapi.responses import RedirectResponse, JSONResponse, Response
# from models import *

import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import json

import os
import numpy as np 
import pandas as pd 

import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization 

import nltk 
from nltk.stem.porter import PorterStemmer 

from sklearn.metrics.pairwise import cosine_similarity 
import warnings
warnings.filterwarnings("ignore")

app = FastAPI()


# In[2]:


data_dir = "data_usaha.csv"
metadata_file = 'synthetic_metadata.json'


# In[3]:



cred = credentials.Certificate("ccusahayuk-firebase-adminsdk-5cq5j-77c8c8f8a8.json")
firebase_admin.initialize_app(cred)

db = firestore.client()
users={}

route = ["users"]


# In[4]:


#Melakukan extract dari csv menjadi dictionary ke firestore 
def extract_from_csv(data_dir):
    data = {
        "user_id":"", 
        "jenis_usaha":"", 
        "skala_usaha":"", 
        "modal_usaha":"", 
        "bidang_usaha":"", 
        "omset_usaha":"", 
        "usia_targetpelanggan":[], 
        "gender_targetpelanggan":[], 
        "pekerjaan_targetpelanggan":[], 
        "status_targetpelanggan":[], 
        "jenis_lokasi_":[]
    }

    df = pd.read_csv(data_dir, index_col=None)
    records = df.to_dict(orient='records')
    print(type(records))
    for i in records:
        data["user_id"] = ""
        data["jenis_usaha"] = ""
        data["skala_usaha"] = ""
        data["modal_usaha"] = ""
        data["bidang_usaha"] =""
        data["omset_usaha"] = ""
        data["usia_targetpelanggan"]=[]
        data["gender_targetpelanggan"]=[]
        data["pekerjaan_targetpelanggan"]=[]
        data["status_targetpelanggan"]=[]
        data["jenis_lokasi_"]=[]
        
        
        user_id = i["user_id"]
        jenis_usaha = i["jenis_usaha"]
        skala_usaha = i["skala_usaha"]
        modal_usaha = i["modal_usaha"]
        bidang_usaha = i["bidang_usaha"]
        omset_usaha = i["omset_usaha"]
        usia_targetpelanggan = i["usia_targetpelanggan"]
        gender_targetpelanggan = i["gender_targetpelanggan"]
        pekerjaan_targetpelanggan = i["pekerjaan_targetpelanggan"]
        status_targetpelanggan = i["status_targetpelanggan"]
        jenis_lokasi_ = i["jenis_lokasi "]

        data["user_id"] = user_id
        data["jenis_usaha"] = jenis_usaha
        data["skala_usaha"] = skala_usaha
        data["modal_usaha"] = modal_usaha
        data["bidang_usaha"] = bidang_usaha
        data["omset_usaha"] = omset_usaha
        data["usia_targetpelanggan"].extend((usia_targetpelanggan.replace("[", "").replace("]", "").replace("'", "")).split(",")),
        data["gender_targetpelanggan"].extend((gender_targetpelanggan.replace("[", "").replace("]", "").replace("'", "")).split(",")), 
        data["pekerjaan_targetpelanggan"].extend((pekerjaan_targetpelanggan.replace("[", "").replace("]", "").replace("'", "")).split(",")),
        data["status_targetpelanggan"].extend((status_targetpelanggan.replace("[", "").replace("]", "").replace("'", "")).split(",")), 
        data["jenis_lokasi_"].extend((jenis_lokasi_.replace("[", "").replace("]", "").replace("'", "")).split(",")),

        print(data)
        route =[
            "users",
            user_id
        ]    
        pushDataCSVtoFirebase(db,route,data)

# Firestore
def pushDataCSVtoFirebase(db,route,data):
    db.collection(route[0]).document(str(route[1])).set(data)
    print("push berhasil")


# In[5]:


def extract_DF_from_firestore(): 
    collection_ref = db.collection('users')
    docs = collection_ref.get()
    
    data = []
    for doc in docs:
        data.append(doc.to_dict())
        
    df = pd.DataFrame(data, columns=['user_id', 'jenis_usaha', 'skala_usaha', 'modal_usaha', 'bidang_usaha', 'omset_usaha', 'usia_targetpelanggan', 'gender_targetpelanggan', 'pekerjaan_targetpelanggan', 'status_targetpelanggan', 'jenis_lokasi_'])
    
    #users = extract_user_datas(df1)
    #Transform
    df.user_id= df.user_id.astype(str)

    #Load
    #load main datas to new dataframe will be used
    df['features'] = df['skala_usaha'].astype(str) + " " + df['modal_usaha'].astype(str) +" "+ df['bidang_usaha'].astype(str) +" "+ df['omset_usaha'].astype(str) +" "+ df['usia_targetpelanggan'].astype(str) +" "+ df['gender_targetpelanggan'].astype(str) +" "+ df['pekerjaan_targetpelanggan'].astype(str) +" "+ df['status_targetpelanggan'].astype(str) +" "+ df['jenis_lokasi_'].astype(str)
    new_df = df[["user_id" ,"features"]]
    new_df.features = new_df.features.apply(lambda x: x.replace(".",""))
    new_df.features = new_df.features.apply(lambda x: x.replace(",","").replace("-","").replace(">","").replace("/"," ").replace("[", "").replace("]", ""))
    new_df.features = new_df.features.apply(lambda x: x.lower())
    # print(new_df.to_string())


    # # new_df
    return df,new_df,users


# In[6]:


#FUNCTION FOR PROTERSTEMMER
def stem(txt):
    y = []
    porterStemmer = PorterStemmer()
    for t in txt.split():
        y.append(porterStemmer.stem(t))
    return " ".join(y)

    """
    input:
        text
    deskripsi:
        melakukan porterstemmer terhadap seluruh kata 
        dalam text dengan menghapus awalan dan akhiran
    output:
        text berisi kata dasar
    """
    
    
#DEFINE TENSORFLOW FOR TEXTVECTORIZATION
def TextVectorize(y):
    text_features= tf.data.Dataset.from_tensor_slices(y)
    max_features = 5000 
    max_len = 70
    vectorize_layer = tf.keras.layers.TextVectorization(
        max_tokens=max_features, 
        output_mode='int', 
        output_sequence_length = max_len)
    vectorize_layer.adapt(text_features.batch(10))
    
    model = tf.keras.models.Sequential()
    model.add(tf.keras.Input(shape=(1,), dtype=tf.string))
    model.add(vectorize_layer)
    x = model.predict(y)
    return x
    

#CALCULATE SIMILARITY 
def getSimilarityMatrix():
    new_df.features = new_df.features.apply(stem)
    y =  TextVectorize(new_df.features)
    similarity = cosine_similarity(y)

    """
    input:
        text
    deskripsi:
        melakukan textvectorization menggunakan tensorflow
        dari hasil output text yang telah di stem
    output:
        tingkat similaritas
    """
    

    return similarity

#FUNCTION FOR GET RECOMMENDATION 
def getRecommendation(cekid):
    
    usaha_index = new_df[new_df['user_id'] == 'nan'].index[0]
    similarity = getSimilarityMatrix()
    distances = similarity[usaha_index]
    usaha_list = sorted(list(enumerate(distances)),reverse=True, key=lambda x:x[1])[1:6]
    
    result = []
    for i in usaha_list:
        item = df.iloc[i[0]].jenis_usaha
        result.append(item)
    return result

        #result.append(df.iloc[i[0]].jenis_usaha)
        #print(result)
        #print(df.iloc[i[0]].modal_usaha, df.iloc[i[0]].bidang_usaha, df.iloc[i[0]].omset_usaha, df.iloc[i[0]].usia_targetpelanggan, df.iloc[i[0]].gender_targetpelanggan) #df.iloc[i[0]].pekerjaan_targetpelanggan,df.iloc[i[0]].jenis_lokasi_ )
        #print(new_df.iloc[i[0]].user_id, " - ", df.iloc[i[0]].jenis_usaha)
        

    """
    input:
        user_id
    deskripsi:
        melakukan perankingan berdasarkan tingkat 
        similaritas antar features milik 
        user_id dengan user_id lainnya
    output:
        rekomendasi usaha berdasarkan tingkat similaritas
    """


# In[7]:


import logging

def to_getID():
    # Dapatkan referensi koleksi yang ingin Anda cari
    collection_ref = db.collection('users')

    # Item data yang diketahui
    user_id = df['user_id'].iloc[-1]
    jenis_usaha = df['jenis_usaha'].iloc[-1]
    skala_usaha = df['skala_usaha'].iloc[-1]
    modal_usaha = df['modal_usaha'].iloc[-1]
    bidang_usaha = df['bidang_usaha'].iloc[-1]
    omset_usaha = df['omset_usaha'].iloc[-1]
    usia_targetpelanggan = df['usia_targetpelanggan'].iloc[-1]
    gender_targetpelanggan = df['gender_targetpelanggan'].iloc[-1]
    pekerjaan_targetpelanggan = df['pekerjaan_targetpelanggan'].iloc[-1]
    status_targetpelanggan = df['status_targetpelanggan'].iloc[-1]
    jenis_lokasi = df['jenis_lokasi_'].iloc[-1]

    # skala_usaha = df.loc[df['user_id'] == '95020165', 'skala_usaha'].iloc[0]
    # modal_usaha = df.loc[df['user_id'] == '95020165', 'modal_usaha'].iloc[0]
    # bidang_usaha = df.loc[df['user_id'] == '95020165', 'bidang_usaha'].iloc[0]
    # omset_usaha = df.loc[df['user_id'] == '95020165', 'omset_usaha'].iloc[0]
    # usia_targetpelanggan = df.loc[df['user_id'] == '95020165', 'usia_targetpelanggan'].iloc[0]
    # gender_targetpelanggan = df.loc[df['user_id'] == '95020165', 'gender_targetpelanggan'].iloc[0]
    # pekerjaan_targetpelanggan = df.loc[df['user_id'] == '95020165', 'pekerjaan_targetpelanggan'].iloc[0]
    # status_targetpelanggan = df.loc[df['user_id'] == '95020165', 'status_targetpelanggan'].iloc[0]
    # jenis_lokasi = df.loc[df['user_id'] == '95020165', 'jenis_lokasi_'].iloc[0]

    # Eksekusi query dan temukan dokumen yang cocok
    # query_ref = collection_ref.where('user_id', '==', 'nan').where('jenis_usaha', '==', 'Nan=N')
    query_ref = collection_ref.where('skala_usaha', '==', skala_usaha).where('modal_usaha', '==', modal_usaha).where('bidang_usaha', '==', bidang_usaha).where('omset_usaha', '==', omset_usaha).where('usia_targetpelanggan', '==', usia_targetpelanggan).where('gender_targetpelanggan', '==', gender_targetpelanggan).where('pekerjaan_targetpelanggan', '==', pekerjaan_targetpelanggan).where('status_targetpelanggan', '==', status_targetpelanggan).where('jenis_lokasi_', '==', jenis_lokasi)
    docs = query_ref.stream()



    doc_id = []
    for doc in docs:
        # Dapatkan ID dokumen yang cocok
        doc_id = doc.id
        # Mencetak daftar ID dokumen
    return str(doc_id).replace("[","").replace("]","").replace("'","")
    
    # doc_ids = []
    # for doc in docs:
        # Dapatkan ID dokumen yang cocok
        # doc_ids.append(doc.id)

    # if doc_ids:
        # return doc_ids[0]
    # else:
        # return None

def pushtoFireStore(): 
    get_id = to_getID()
    if get_id is not None: 
        get_id = get_id.rstrip("/")
        collection_ref1 = db.collection("users")
        document_ref = collection_ref1.document(get_id)

        # Data yang akan diupdate
        data = {
            "Hasil_rekomendasi": result,
            "jenis_usaha": result[0],
            "user_id": get_id
        }

        # Melakukan update pada dokumen
        document_ref.update(data)
        logger = logging.getLogger()
        fhandler = logging.FileHandler(filename='test_log.log', mode='a')
        logger.addHandler(fhandler)
        logging.warning(data)
        hasil = "Berhasil, berikut hasil rekomendasi untuk anda : " + str(result)
        return hasil
    else:
        return "Silakan coba lagi"


# In[8]:


df, new_df, users = extract_DF_from_firestore()


# In[9]:


df


# In[10]:


cekid = new_df[new_df['user_id'] == 'NaN']
result = getRecommendation(cekid)


# In[11]:


result


# In[12]:


pushtoFireStore()


# In[13]:


@app.get("/")
def root(): 
    return RedirectResponse("http://127.0.0.1:8000/docs")


@app.post("/recommendation/")
def get_recommendation():
    global df, new_df, users
    df, new_df, users = extract_DF_from_firestore()
    cekid = new_df[new_df['user_id'] == 'nan']
    result = getRecommendation(cekid)
    pushtoFireStore()
    json_recommendations = json.dumps(result) 
    return json_recommendations 


# In[14]:


import uvicorn
import asyncio

async def run_server():
    config = uvicorn.Config(app)
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == "__main__":
    asyncio.run(run_server())
