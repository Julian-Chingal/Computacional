# library
from moviepy.editor import VideoFileClip
from moviepy.editor import CompositeVideoClip
from moviepy.editor import TextClip
from moviepy.config import change_settings
from moviepy.video.tools.subtitles import file_to_subtitles, SubtitlesClip
import subprocess
import tkinter as tk
from tkinter import filedialog
import whisper

change_settings({"IMAGEMAGICK_BINARY": r"C:/Program Files/ImageMagick-7.1.1-Q16-HDRI/magick.exe"})

# Variables
model = whisper.load_model("small")
audio_output_path = "Entrega_3/Data/audio_video.wav"
subtitles_path = "Entrega_3/Data/translate/audio_video.srt"
l = 'en'

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
    # video_clip.close()

    transcribe_audio(video_path,video_clip)

def transcribe_audio(video_path,video_clip):
    print("\nObteniendo la transcripcion y traduccion del video ...........")

    # command = f'whisper "{audio_output_path}" --model small --language "{l}" --output_dir "Entrega_3/Data/translate"'
    # subprocess.run(command, shell=True)

    print("Agregando subtitulos ......")

    srt_subtitles = file_to_subtitles(subtitles_path) # Crear el clip de subtítulos desde el archivo
    generator = lambda txt: TextClip(txt, font='Arial', fontsize=18, color='white')
    subtitles = SubtitlesClip(srt_subtitles, generator)
    result = CompositeVideoClip([video_clip, subtitles.set_position(('center', 'bottom'))])

    result.write_videofile('Entrega_3/Data/video_con_subtitulos.mp4', codec='libx264', audio_codec='aac') # Guardar el video resultante

#! init -------------------------------------------------------------
select_video()