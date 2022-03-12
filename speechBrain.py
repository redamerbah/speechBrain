from asyncio import sleep
import speech_recognition as sr

from speechbrain.pretrained import SpeakerRecognition
import torchaudio

import os

os.system("rm *.wav")

testeur = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_models/spkrec-ecapa-voxceleb")

recognizer = sr.Recognizer()

with sr.Microphone() as source:
    print("Gardez le silence pour un moment svp !")
    recognizer.adjust_for_ambient_noise(source=source, duration=10)
    print("parler... !!!")
    try:
        enregistrement = recognizer.listen(source)

        fichier = open("./tempo.wav", "wb")
        fichier.write(enregistrement.get_wav_data())
        fichier.close()

        score, prediction = testeur.verify_files("./tempo.wav", "./personne_enregistree/testReda.wav")
        if(prediction == True):
            print("attendez un peu svp")
            text = recognizer.recognize_google(enregistrement, language="fr-FR")
            print("Vous avez dit : " + text)
        else:
            print("Nous n'avons pas pu vous identifier, veuillez reessayer plus tard.")

    except:
        print("erreur")

