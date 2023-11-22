import requests

# Variables
url = "https://api-car-dev-dkkz.4.us-1.fl0.io/api"

# Metodo Get para todas las rutas de la API
def get(endpoint):
    try:
        _url = f'{url}/{endpoint}'
        response = requests.get(_url)
        print(f'{url}/{endpoint}')
        if response.status_code == 200:
            data = response.json()
            print("Datos obtenidos (GET) 🎉")
            return data
        else:
            print("Error en la petición (GET)", response.status_code)
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error en la solicitud 🥶: {e}")
        return None

# Metodo Post para todas las rutas de la API
def post(endpoint, data):
    try:
        _url = f'{url}/{endpoint}'
        response = requests.post(_url, json=data)    

        if response.status_code == 200:
            data = response.json()
            print("Datos obtenidos (POST) 🎉")
            return data
        else:
            print("Error en la petición (POST)", response.status_code)
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error en la solicitud 🥶: {e}")
    response = requests.post(f'{url}/{endpoint}', json=data)

# Metodo Delete para todas las rutas de la API
def delete(endpoint):
    try:
        _url = f'{url}/{endpoint}'
        response = requests.delete(_url)    

        if response.status_code == 200:
            data = response.json()
            print("Tablas formateadas 🎉")
            return data
        else:
            print("Error en la petición (DELETE)", response.status_code)
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error en la solicitud 🥶: {e}")
