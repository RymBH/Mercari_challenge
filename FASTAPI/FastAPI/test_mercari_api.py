import requests
from pydantic import BaseModel
# ________________________________________Tests routes API____________________________________

class Item(BaseModel):
    name: str
    item_condition_id: int
    category_name: str
    brand_name: str
    shipping: int
    item_description: str

# URL API
BASE_URL = 'http://localhost:8000'

# Test root
def test_read_root():
    response = requests.get(f'{BASE_URL}/')
    assert response.status_code == 200
    assert response.json() == {"Predict": "Mercari item price"}

# Test bdd utilisateurs
def test_get_users():
    response = requests.get(f'{BASE_URL}/vendors')
    assert response.status_code == 200
    assert isinstance(response.json(), dict)

# Test statut utilisateur
def test_get_status():
    auth = ('Eleonora', 'builder') 
    response = requests.get(f'{BASE_URL}/status', auth=auth)
    assert response.status_code == 200
    assert response.json() == 1

# Test prediction avec utilisateur authentifié
def test_get_prediction():
    auth = ('Eleonora', 'builder')
    mercari_item = Item(name='J Lo Glow Perfume', 
                        item_condition_id=3, 
                        category_name='Beauty/Fragrance/Women',
                        brand_name='Sephora',
                        shipping=0,
                        item_description='3/4 bottle of JLo Glow Big Bottle')
    headers = {'Content-Type': 'application/json'}
    response = requests.post(f'{BASE_URL}/prediction', auth=auth, headers=headers, data=mercari_item.json())
    assert response.status_code == 200

# Test prediction avec utilisateur non authentifié
def test_get_prediction_unauthorized():
    auth = ('unknown', 'unknown')
    mercari_item = Item(name='J Lo Glow Perfume', 
                        item_condition_id=3, 
                        category_name='Beauty/Fragrance/Women',
                        brand_name='Sephora',
                        shipping=0,
                        item_description='3/4 bottle of JLo Glow Big Bottle')
    headers = {'Content-Type': 'application/json'}
    response = requests.post(f'{BASE_URL}/prediction', auth=auth, headers=headers, data=mercari_item.json())
    assert response.status_code == 401
    assert response.json() == {"detail": "Incorrect email or password"}