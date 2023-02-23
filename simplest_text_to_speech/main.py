import speech_recognition as sr

r = sr.Recognizer()
FILENAME = 'audio.wav'
with sr.AudioFile(FILENAME) as source:
    audio = r.listen(source)

    try:
        print('Converting audio transcripts into text ...')
        text = r.recognize_google(audio)
        print('You said: {}'.format(text))
    except:
        print('Error: Could not convert audio into text')
