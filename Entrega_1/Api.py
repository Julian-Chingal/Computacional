import requests
from queue import Queue
import threading

base_url = "https://api-car-dev-dkkz.4.us-1.fl0.io/api/"

# Función para hacer una solicitud POST
def make_request_post(endpoint, data):
    try:
        url = f"{base_url}{endpoint}"

        response = requests.post(url, json=data)

        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error en la solicitud. Código de respuesta: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error en la solicitud: {e}")
        return None

def make_request_delete(endpoint):
    try:
        url = f"{base_url}{endpoint}"
        
        response = requests.delete(url)
        
        if response.status_code == 200:
            data = response.json()
            return data
        else:
            print(f"Error en la solicitud. Código de respuesta: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error en la solicitud: {e}")
        return None

# Función principal para enviar trayectoria usando un hilo
def post_route(data):
    endpoint = "route"

    make_request_delete(endpoint)

    # Crear una cola para comunicarse con el hilo
    cola_trayectoria = Queue()

    # Crear el hilo y configurar el target
    hilo_enviar_trayectoria = threading.Thread(target=enviar_trayectoria, args=(endpoint, cola_trayectoria))

    # Iniciar el hilo
    hilo_enviar_trayectoria.start()

    # Enviar trayectoria del AG a la cola (Hilo)
    for punto in data:
        cola_trayectoria.put(punto)

    # Indicar que no hay más datos en la cola y esperar a que el hilo termine
    cola_trayectoria.put(None)
    hilo_enviar_trayectoria.join()

    # Obtener el resultado de la función make_request_post
    resultado_post_route = cola_trayectoria.get()

    return resultado_post_route

# Función que ejecuta el hilo y realiza la solicitud POST
def enviar_trayectoria(endpoint, cola_trayectoria):
    data = []
    while True:
        punto = cola_trayectoria.get()
        if punto is None:
            break
        x, y, x2, y2, cm, degree = punto
        data.append({"x_route": x, "y_route": y,"cm_route": 20, "degree_route": degree, "state": True})  # Ajusta según la estructura real

    # Realizar la solicitud POST usando la función make_request_post
    resultado_post_route = make_request_post(endpoint, data)

    # Colocar el resultado en la cola para que pueda ser obtenido por la función principal
    cola_trayectoria.put(resultado_post_route)