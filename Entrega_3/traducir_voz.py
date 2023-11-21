# Librery
import speech_recognition as sr
from googletrans import Translator
import pyaudio as pa
import threading

# Varibles
sample_rate = 44100
chunk_size = 1024

# start 
p = pa.PyAudio()
default_device_index = p.get_default_input_device_info().get('index')
default_device_name = p.get_device_info_by_index(default_device_index).get('name')

r = sr.Recognizer()
trans = Translator()
mic = sr.Microphone(sample_rate= sample_rate, chunk_size= chunk_size)

print(f'\nMicrofono utilizado ${default_device_name}')

# Utilizar el micrófono como fuente de audio
def recognize_speech():
    with mic as source:
        r.adjust_for_ambient_noise(source)
        print("Di algo... ")

        while True: 
            try:
                audio = r.listen(source, timeout= 0.5)  
                text = r.recognize_google(audio, language='es-ES')
                print(f'\ntext: {text}')

                # Traducir el texto al inglés
                translation = trans.translate(text, src='es', dest='pt')
                print(f'(English): {translation.text}')
            except sr.WaitTimeoutError:
                print("tiempo de espera agotado")
            except sr.UnknownValueError:
                pass  
            except sr.RequestError as e:
                print("Error al solicitar resultados: {0}".format(e))
            

# Crear un hilo para reconocer el habla
speech_thread = threading.Thread(target=recognize_speech)
speech_thread.start()