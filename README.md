# Recommender System Implementation of Tensorflow and Scikit Learn
1. The model adapted from [Kaggle Recommendation System](https://www.kaggle.com/code/sagarbapodara/coursera-course-recommendation-system-webapp)
2. Here is our dataset [Dataset Usahayuk](https://github.com/Usahayuk-Capstone-Project/Machine-Learning/blob/main/data_usaha.csv)

Recommender systems is used in our applications to provide personalized recommendations to users. Two common approaches used in recommender systems are content-based filtering and collaborative filtering. In this project we are using Content-Based Filtering, because its focuses on the attributes or characteristics of the items being recommended. It recommends items to users based on their preferences for certain item features. The system analyzes the content or metadata of the items, in their input data (text), and matches them with the user's preferences. 

### Import library and dependencies 
```
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
```
```
app = FastAPI()
data_dir = "data_usaha.csv"
metadata_file = 'synthetic_metadata.json'
cred = credentials.Certificate("ccusahayuk-firebase-adminsdk-5cq5j-77c8c8f8a8.json")
firebase_admin.initialize_app(cred)
db = firestore.client()
users={}
route = ["users"]
```
### Push the data from CSV file to Firestore
```
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
```
## Load Data and Data Preparation 
ETL or Extract, Transform, Load is process to extract data from firebase containing dummy data into a dataFrame. Then, we extract new feature from all the features that are provided as input by the user. Next, we perform data cleaning for characters other than necessary text such as / ' ', > < =.

### Extract the data from Firestore into dataframe
```
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
```
## Cosine Similarity Implementation for Recommendation System
Porter Stemmer is used for cleaning text within the features using the Porter stemmer algorithm. This involves removing prefixes or suffixes from words to reduce the amount of data and expedite the data processing.
After applying the Porter Stemmer, the text within the features is converted from text to numeric representation in the form of vectors. The basic concept is the implementation of word embedding.

### Implementation of PorterStemmer 

```
#FUNCTION FOR PROTERSTEMMER
def stem(txt):
    y = []
    porterStemmer = PorterStemmer()
    for t in txt.split():
        y.append(porterStemmer.stem(t))
    return " ".join(y)
    
```

### Implementation of Tensorflow for TextVectorization 
```
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
    
  ```

 ### Implementation of Scikit Learn for Cosine Similarity 
Cosine Similarity Using Scikit-Learn
The vectorized data will then undergo cosine similarity calculation between one vector and another. Cosine similarity ranges from 0 to 1, where a value closer to 1 indicates a higher similarity between the user input and the reference data.
Next, a ranking is performed for each level of similarity. We take the top 5 results with the highest similarity level.
   ```
   #CALCULATE SIMILARITY 
def getSimilarityMatrix():
    new_df.features = new_df.features.apply(stem)
    y =  TextVectorize(new_df.features)
    similarity = cosine_similarity(y)
    return similarity
#FUNCTION FOR GET RECOMMENDATION 
def getRecommendation(cekid):
    usaha_index = new_df[new_df['user_id'] == ''].index[0]
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
```
## Return the Result to Update the Data in Firestore
### First, we search the documentID 

```
def to_getID():
    # Get the reference to the collection you want to query
    collection_ref = db.collection('users')
    # Known data items
    skala_usaha = df.loc[df['user_id'] == "", 'skala_usaha'].values[0]
    modal_usaha = df.loc[df['user_id'] == "", 'modal_usaha'].values[0]
    bidang_usaha = df.loc[df['user_id'] == "", 'bidang_usaha'].values[0]
    omset_usaha = df.loc[df['user_id'] == "", 'omset_usaha'].values[0]
    usia_targetpelanggan = df.loc[df['user_id'] == "", 'usia_targetpelanggan'].values[0]
    gender_targetpelanggan = df.loc[df['user_id'] == "", 'gender_targetpelanggan'].values[0]
    pekerjaan_targetpelanggan = df.loc[df['user_id'] == "", 'pekerjaan_targetpelanggan'].values[0]
    status_targetpelanggan = df.loc[df['user_id'] == "", 'status_targetpelanggan'].values[0]
    jenis_lokasi = df.loc[df['user_id'] == "", 'jenis_lokasi_'].values[0]
    user_id = df.loc[df['user_id'] == "", 'user_id'].values[0]
    jenis_usaha = df.loc[df['user_id'] == "", 'jenis_usaha'].values[0]
    
    # Execute the query and find matching documents
    query_ref = collection_ref.where('user_id', '==', user_id).where('jenis_usaha', '==', jenis_usaha).where('skala_usaha', '==', skala_usaha).where('modal_usaha', '==', modal_usaha).where('bidang_usaha', '==', bidang_usaha).where('omset_usaha', '==', omset_usaha).where('usia_targetpelanggan', '==', usia_targetpelanggan).where('gender_targetpelanggan', '==', gender_targetpelanggan).where('pekerjaan_targetpelanggan', '==', pekerjaan_targetpelanggan).where('status_targetpelanggan', '==', status_targetpelanggan).where('jenis_lokasi_', '==', jenis_lokasi)
    doc_ids = []
    for doc in query_ref.stream():
        # Get the ID of each matching document
        doc_ids.insert(0, doc.id)  # Insert the new ID at the beginning of the list
    return doc_ids
returned_doc_ids = to_getID()
# Print the first document ID
if len(returned_doc_ids) > 0:
    print(returned_doc_ids[0])
else:
    print("No matching documents found.")
```
### Second, we update the field of data into the document 
```
def pushtoFireStore(): 
    get_id = str(to_getID()).replace("]","").replace("[","").replace("'","")
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
        
 ```
 ## FastAPI Implementation for Generate Model API
 FastAPI is used to build APIs so that models can be utilized in mobile development. By using FastAPI, we can develop APIs that can be integrated into mobile applications. By utilizing FastAPI to build your API, you can separate the machine learning model logic from the mobile app development, making it easier to maintain and update the model independently. The mobile app can then make API requests to the FastAPI endpoints to leverage the model's capabilities.
 ```
 @app.get("/")
def root(): 
    return RedirectResponse("http://127.0.0.1:8000/docs")
@app.post("/recommendation/")
def get_recommendation():
    global df, new_df, users
    df, new_df, users = extract_DF_from_firestore()
    cekid = new_df[new_df['user_id'] == '']
    result = getRecommendation(cekid)
    pushtoFireStore()
    json_recommendations = json.dumps(result) 
    return json_recommendations 
```

### 
```
import uvicorn
import asyncio
async def run_server():
    config = uvicorn.Config(app)
    server = uvicorn.Server(config)
    await server.serve()
if __name__ == "__main__":
    asyncio.run(run_server())
```
