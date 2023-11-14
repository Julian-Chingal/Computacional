# library
from moviepy.editor import VideoFileClip
from moviepy.video.tools.subtitles import SubtitlesClip
import subprocess
import tkinter as tk
from tkinter import filedialog
import whisper
from googletrans import Translator

# Variables
model = whisper.load_model("small")
trans = Translator()
audio_output_path = "Entrega_3/Data/audio_video.wav"
subtitles_output_path = "Entrega_3/Data/subtitles.txt"

#! Functions -------------------------------------------------------
def select_video():
    root = tk.Tk()
    root.withdraw()
    video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4")])
    if video_path:
        extraer_audio(video_path)

def extraer_audio(video_path):
    video_clip = VideoFileClip(video_path)
    audio_clip = video_clip.audio
    audio_clip.write_audiofile(audio_output_path, codec='mp3')
    video_clip.close()

    transcribe_audio(video_path)

def transcribe_audio(video_path):
    print("\nObteniendo la transcripcion del video ...........")
    text = model.transcribe(audio_output_path)
    translation = trans.translate(text["text"], src='es', dest='en')
    print(text["text"])
    print(f'\nTraduccion: \n{translation.text}')

    #Guardar los subtítulos 
    with open(subtitles_output_path, 'w', encoding='utf-8') as writefile:
        writefile.write(text["text"]) 
        writefile.write(translation.text)

#! init -------------------------------------------------------------
select_video()