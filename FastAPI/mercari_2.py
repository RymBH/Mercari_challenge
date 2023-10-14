import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
import pandas as pd
import numpy as np
import scipy
import scipy.sparse
from joblib import load
import os
import nltk
from nltk.corpus import stopwords
#nltk.download('stopwords')
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.metrics import mean_squared_error as mse
from fastapi import FastAPI
from pydantic import BaseModel
import json
import uvicorn
from fastapi import Depends, FastAPI
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi import FastAPI, HTTPException,status
from pydantic import BaseModel

path = '/Users/rymbenhassine/Documents/MLops/Mercari/fastapi/'
#Loading data
X_train = scipy.sparse.load_npz(path+"train_final.npz")
y_train = np.load(path+"y_train.npy")

X_test = scipy.sparse.load_npz(path+"test_final.npz")

#créé le df dans lequel chercher meracri_item
#item_df = pd.read_csv(path+'train.tsv', sep='\t')
item_df = pd.read_csv(path+'test.tsv', sep='\t')

#importe le modèle
model_svr = load('model_svr.joblib')

# fonction pour historique.json
def update_sales_history(sale_item: dict, filename: str):
    try:
        with open(filename, 'r') as f:
            sales_history = json.load(f)
    except FileNotFoundError:
        sales_history = []

    sales_history.append(sale_item)

    with open(filename, 'w') as f:
        json.dump(sales_history, f)


api = FastAPI(
    title='Mercari',
    description="Price suggestion",
        version="1.0.1",
    #openapi_tags=[
    #{
    #    'name': 'Home',
    #    'description': 'Default functions'
    #},
    #{
    #    'name': 'Prediction',
    #    'description': 'Functions that return predictions'
    #}
    #]
)

@api.get("/",tags=['Vendors'])
def read_root():
    return {"Predict": "Mercari item price"}

security = HTTPBasic()
  


users = {
  "Amina": "wonderland",
  "Eleonora": "builder",
  "Rym": "mandarine"
}


class Item(BaseModel):
    name: str
    item_condition_id: int
    category_name: str
    brand_name: str
    shipping: int
    item_description: str


def get_current_username(credentials: HTTPBasicCredentials = Depends(security)):
    for key, value in users.items():
        if credentials.username==key and credentials.password==value:
            return credentials.username

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Incorrect email or password",
        headers={"WWW-Authenticate": "Basic"},
    )

def get_admin_username(credentials: HTTPBasicCredentials = Depends(security)):
    if credentials.username=='admin' and credentials.password=='4dm1N':
        return credentials.username

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Incorrect email or password",
        headers={"WWW-Authenticate": "Basic"},
    )

@api.get('/vendors', name='Get vendors', tags=['Vendors'])
def get_users():
    return users

@api.get('/status', name='Get API status', tags=['Vendors'])
def get_status(username: str = Depends(get_current_username)):
    """Cette fonction renvoie 1 si l'API fonctionne.
    """
    return 1

@api.post("/prediction", name='Get prediction', tags=['Sales'])
async def get_prediction(mercari_item: Item, username: str = Depends(get_current_username)):


    #prediction

    item_data = item_df[(item_df['name'] == mercari_item.name) & 
                        (item_df['item_condition_id'] == mercari_item.item_condition_id) & 
                        (item_df['category_name'] == mercari_item.category_name) & 
                        (item_df['brand_name'] == mercari_item.brand_name) & 
                        (item_df['shipping'] == mercari_item.shipping)]
    
    print(mercari_item)
    print(item_data)

    if  not item_data.empty:
        item_index = item_data.index[0]
    else :
        raise HTTPException(status_code=400, detail="Item not found")

    if item_index < X_test.shape[0]:
        item_features = X_test[item_index]
        item_features = item_features.reshape(1, -1)
        prediction_svr = model_svr.predict(item_features)

        print(prediction_svr)

    else :
        raise HTTPException(status_code=400, detail="Item not found")
    

    prediction = prediction_svr

    # If the prediction is not valid then we raise an error
    if prediction < 0:
        raise HTTPException(status_code=400, detail="Invalid prediction")
    else :
        mercari_item_dict = mercari_item.dict()
        prediction = np.expm1(prediction)
        prediction = int(prediction[0])
        mercari_item_dict['predicted_price'] = prediction
        # update the sales history
        update_sales_history(mercari_item_dict, path + 'historique.json')

    return {"Predicted Price": prediction, }

item_condition_db= [{'condition': '1'},
                    {'condition': '2'},
                    {'condition': '3'},
                    {'condition': '4'},
                    {'condition': '5'}]

shipping_db =[{'shipping': '1'},
              {'shipping': '0'}]

def read_sales_history(filename: str):
    try:
        with open(filename, 'r') as f:
            sales_history = json.load(f)
    except FileNotFoundError:
        sales_history = []

    return sales_history

historique_db = read_sales_history(path + 'historique.json')

#@api.post ("/sales/", tags= ["Sales"])
#async def post_sale(input_data: Item):
#    historique_db.append(input_data)
#    return input_data

# Check if file exists before writing to it for the first time
if not os.path.exists(path + 'historique.json'):
    with open(path + 'historique.json', 'w') as f:
        json.dump(historique_db, f)



@api.get("/sales/historique", description=" all sales",tags= ["Sales"])
async def get_historique():
    historique_db = read_sales_history(path + 'historique.json')
    return historique_db


if __name__ == "__mercari__":
    uvicorn.run(api, port=8000, host="0.0.0.0")