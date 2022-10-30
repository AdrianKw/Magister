import librosa
import os
import pickle
import random
import numpy as np
import pandas as pd
import speech_recognition as sr
from python_speech_features import mfcc
from scipy.io import wavfile
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from kivy.app import App
from kivy.clock import Clock
from kivy.properties import StringProperty
from kivy.uix.screenmanager import ScreenManager, Screen


def trim_silence_and_normalization(data_wav):
    data_norm = data_wav * (1 / np.max(np.abs(data_wav)))
    clips = librosa.effects.split(data_norm, top_db=20)
    wav_data = []
    for c in clips:
        data_split = data_norm[c[0]: c[1]]
        wav_data.extend(data_split)
    return np.array(wav_data)


def check_personality(dana, samplerate, My_model):
    data_trim_norm = trim_silence_and_normalization(dana)
    mfcc_feat_test = mfcc(data_trim_norm, samplerate, winlen=winlen, winstep=winstep, numcep=numcep,
                          nfilt=nfilt, nfft=nfft, ceplifter=ceplifter, appendEnergy=appendEnergy, winfunc=winfunc)
    df_mfcc_feat_test = pd.DataFrame(scaler_standard.transform(mfcc_feat_test))
    Prediction = pd.DataFrame(My_model.predict(df_mfcc_feat_test))
    hois = Prediction.value_counts().idxmax()[0]
    howmuch = Prediction.value_counts(normalize=True).max() * 100
    return hois, howmuch


def feature_extraction():
    path_train = "obrobka\\"
    final_train = pd.DataFrame()
    file_train_list = os.listdir(path_train)
    for file_train in file_train_list:
        (sr_train, y_train) = wavfile.read(path_train + file_train)
        data_trim_norm = trim_silence_and_normalization(y_train)
        mfcc_feat_train = mfcc(data_trim_norm, sr_train, winlen=winlen, winstep=winstep, numcep=numcep, nfilt=nfilt,
                               nfft=nfft, ceplifter=ceplifter, appendEnergy=appendEnergy, winfunc=winfunc)
        df_mfcc_feat_train = pd.DataFrame(mfcc_feat_train)
        df_mfcc_feat_train['Osoba'] = file_train
        final_train = pd.concat([final_train, df_mfcc_feat_train])
    final_train.reset_index(drop=True)
    final_train.to_csv('Trenujacy.csv', index=False)


def training_algorythm():
    scaler_standard = StandardScaler()
    final_training = pd.read_csv('Trenujacy.csv')
    y = final_training['Osoba']
    x = final_training.drop('Osoba', axis=1)
    x = scaler_standard.fit_transform(x)
    clf = svm.SVC(kernel='rbf', C=0.5, gamma='auto')
    clf.fit(x, y)
    pickle.dump(scaler_standard, open('finalized_scaler.sav', 'wb'))
    pickle.dump(clf, open('finalized_model_svm.sav', 'wb'))


def load_model_svm_and_scaler():
    loaded_model = pickle.load(open('finalized_model_svm.sav', 'rb'))
    loaded_scaler = pickle.load(open('finalized_scaler.sav', 'rb'))
    return loaded_model, loaded_scaler


def check_speaker_words(random_word):
    with sr.Microphone() as source:

        recognizer.energy_threshold = 100
        audio_text = recognizer.listen(source)
        with open("microphone-results.wav", "wb") as f:
            f.write(audio_text.get_wav_data())
        try:
            pass
            print(recognizer.recognize_google(audio_text, language="pl-PL").lower())
            x = recognizer.recognize_google(audio_text, language="pl-PL").lower().split()
            x = " ".join(x[2:])
            if x == random_word:
                return "Tekst rozpoznany i dopasowany pozytywnie", True
            else:
                return "Tekst rozpoznany lecz niedopasowany spróbuj jeszcze raz", False
        except:
            return "Nierozpoznano tekstu, spróbuj jeszcze raz", False


def random_words():
    word1 = random.choice(WORDS)
    word2 = random.choice(WORDS)
    word3 = random.choice(WORDS)
    return word1 + " " + word2 + " " + word3


class Menu(Screen):
    random_word = StringProperty("")
    my_text = StringProperty("Po naciśnieciu przycisku przeczytaj wyrazy")
    jakbardzo = StringProperty("")
    ktoto = StringProperty("")

    def get_word(self, _):
        self.my_text = "Proszę mówić"
        self.random_word = random_words()
        Clock.schedule_once(self.get_check)

    def get_check(self, _):
        info, whatnow = check_speaker_words(self.random_word)
        self.my_text = info
        self.random_word = ""

        if whatnow:
            self.my_text = f"{self.my_text}\n\n             Szukanie dopasowania w bazie..."
            Clock.schedule_once(self.restof)

    def on_button_click(self):
        Clock.schedule_once(self.get_word)

    def on_button_release(self):
        Clock.unschedule(self.default_value)
        Clock.schedule_once(self.default_value, 10)

    def restof(self, _):
        fs, wavdata = wavfile.read("microphone-results.wav")
        who, howmuch = check_personality(dana=wavdata, samplerate=fs, My_model=Model)
        self.jakbardzo = str(int(howmuch))
        self.ktoto = str(who)[:-4]
        self.ids.circular_progress.progress = int(howmuch) * 3.6
        if howmuch > 80:
            self.my_text = "Dostęp przyznany"
            self.ids.circular_progress.color_progress = (0, 1, 0, 0.8)
        else:
            self.ids.circular_progress.color_progress = (1, 0, 0, 0.8)
            self.my_text = "Niski współcznynnik dopasowania proszę spróbować jeszcze raz"

    def default_value(self, *args):
        self.random_word = ""
        self.my_text = "Po naciśnieciu Przycisku przeczytaj wyrazy"
        self.jakbardzo = ""
        self.ktoto = ""
        self.ids.circular_progress.progress = 0


class SettingsScreen(Screen):
    def extraction(self):
        feature_extraction()

    def training(self):
        training_algorythm()


class TheLabApp(App):
    def build(self):
        sm = ScreenManager()
        sm.add_widget(Menu(name='menu'))
        sm.add_widget(SettingsScreen(name='settings'))

        return sm


WORDS = ["banan", "arbuz", "mango", "wiśnia", "ananas", "gruszka"]

winlen = 0.12
winstep = winlen / 2
numcep = 22
nfilt = 26
nfft = int(2 ** 14)
appendEnergy = False
ceplifter = 22
winfunc = np.hamming

recognizer = sr.Recognizer()
feature_extraction()
training_algorythm()
Model, scaler_standard = load_model_svm_and_scaler()

TheLabApp().run()
