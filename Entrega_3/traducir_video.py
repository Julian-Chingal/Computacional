# library
from moviepy.editor import VideoFileClip
from moviepy.editor import CompositeVideoClip
from moviepy.editor import TextClip
from moviepy.config import change_settings
from moviepy.video.tools.subtitles import file_to_subtitles, SubtitlesClip
import subprocess
import tkinter as tk
from tkinter import filedialog
# import whisper

#! Variables --------------------------------------------------------------------------------------------------------
change_settings({"IMAGEMAGICK_BINARY": r"C:/Program Files/ImageMagick-7.1.1-Q16-HDRI/magick.exe"})

# model = whisper.load_model("small")
audio_output_path = "Entrega_3/Data/audio_video.wav"
subtitles_path = "Entrega_3/Data/translate/audio_video.srt"

#! Functions ----------------------------------------------------------------------------------------------------------
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

    transcribe_audio(video_clip)

def wrap_text(txt, max_width, font='Arial', fontsize=20, color='white'):
    words = txt.split(' ')
    lines = []
    current_line = []
    current_length = 0

    for word in words:
        word_length = TextClip(word, font=font, fontsize=fontsize, color=color).size[0]

        if current_length + word_length > max_width:
            lines.append(' '.join(current_line))
            current_line = [word]
            current_length = word_length
        else:
            current_line.append(word)
            current_length += word_length

    lines.append(' '.join(current_line))

    wrapped_txt = '\n'.join(lines)

    return TextClip(wrapped_txt, font=font, fontsize=fontsize, color=color)

def transcribe_audio(video_clip):
    print("\nIdiomas Disponibles:", 
          "\n-----------------------",
          "\nEspañol = es \nIngles = en \nPortugues = pt")
    l = input("\nDigite el idioma: ")

    print("\nObteniendo la transcripcion y traduccion del video ...........\n")

    command = f'whisper "{audio_output_path}" --model small --language "{l}" --output_dir "Entrega_3/Data/translate"'
    subprocess.run(command, shell=True)

    print("\nAgregando subtitulos ......\n")

    #ancho y alto del video
    video_width, video_height = video_clip.size 

    srt_subtitles = file_to_subtitles(subtitles_path) # Crear el clip de subtítulos desde el archivo
    generator = lambda txt: wrap_text(txt, video_width)
    subtitles = SubtitlesClip(srt_subtitles, generator)

    result = CompositeVideoClip([video_clip, subtitles.set_position(('center', 'bottom'))])

    result.write_videofile('Entrega_3/Data/video_con_subtitulos.mp4', codec='libx264', audio_codec='aac') # Guardar el video 

#! init -------------------------------------------------------------
select_video()