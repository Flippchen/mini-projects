import os
from tkinter import *
from pytube import YouTube


def download_video():
    url = url_entry.get()
    resolution = resolution_var.get()
    download_folder = "downloads"
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)

    try:
        yt = YouTube(url)
        stream = yt.streams.filter(res=resolution, progressive=True, file_extension='mp4').first()
        if not stream:
            download_label.config(text=f"No stream available for {resolution} resolution.")
            return
        download_label.config(text=f"Downloading {yt.title}...")
        stream.download(download_folder)
        download_label.config(text=f"Download complete! Saved in {download_folder} folder.")
    except Exception as e:
        download_label.config(text=f"Error: {str(e)}")


root = Tk()
root.title("YouTube Video Downloader")

url_label = Label(root, text="YouTube Video URL:")
url_label.pack(pady=10)

url_entry = Entry(root, width=50)
url_entry.pack(pady=10)

# Drop-down menu for resolution selection
resolution_var = StringVar(root)
resolution_var.set("720p")
resolutions = ["144p", "240p", "360p", "480p", "720p", "1080p"]
resolution_menu = OptionMenu(root, resolution_var, *resolutions)
resolution_menu.pack(pady=10)

download_button = Button(root, text="Download Video", command=download_video)
download_button.pack(pady=20)

download_label = Label(root, text="")
download_label.pack(pady=10)

root.mainloop()
