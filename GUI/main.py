import os
import numpy as np
import librosa
import pandas as pd
from python_speech_features import mfcc
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn import svm
from scipy.io import wavfile
from sklearn.model_selection import cross_val_score
import soundfile as sf


def calculate_metrics(target, prediction, average='weighted'):
    accuracy = accuracy_score(target, prediction)
    precision = precision_score(target, prediction, average=average)
    recall = recall_score(target, prediction, average=average)
    f1 = f1_score(target, prediction, average=average)
    mislabeled = (target != prediction).sum()
    total = len(target)
    return accuracy, precision, recall, f1, mislabeled, total


def print_results(metrics, classifier_id='classifier'):
    print(f'Results for {classifier_id}')
    print('----')
    print(f'  Accuracy:  {metrics[0] * 100:.4f} %')
    print(f'  Precision: {metrics[1] * 100:.4f} %')
    print(f'  Recall:    {metrics[2] * 100:.4f} %')
    print(f'  F1 score:  {metrics[3] * 100:.4f} %')
    print(f'  Mislabeled {metrics[4]} out of {metrics[5]}')
    print('\n')


def print_min_max(collumn):
    print(f'  Maksimum:  {collumn.max():.4f}')
    print(f'  Minimum:   {collumn.min():.4f}')
    print(f'  Średnia:   {collumn.mean():.4f}')
    print(f'  Mediana:   {collumn.median():.4f}')


def Model_poza(My_model, plik_csv):
    final_poza = pd.DataFrame()
    Thatistrue_poza = pd.DataFrame()
    for file in plik_csv['Name'].unique():
        One_person_poza = plik_csv.loc[plik_csv['Name'] == file].drop('Name', axis=1)
        One_person_poza = scaler_standard.transform(One_person_poza)

        Prediction_poza = My_model.predict(One_person_poza)  # tutaj zamieniamy knn na inny model do predykcji

        # Obliczanie na ile osoba sprawdzana jest podobna do samej siebie
        df11_poza = pd.DataFrame(Prediction_poza, columns=['Ho'])
        df = pd.DataFrame({'Name': file}, index=[0])
        final_poza = pd.concat([final_poza, df]).reset_index(drop=True)

        # Obliczanie do kogo osoba sprawdzana jest najbardziej podobna
        hois = df11_poza.value_counts().idxmax()[0]
        howmuch = df11_poza.value_counts(normalize=True).max() * 100
        datahow = pd.DataFrame({'Similar to': hois, 'How much [%]': howmuch}, index=[0])
        Thatistrue_poza = pd.concat([Thatistrue_poza, datahow]).reset_index(drop=True)
        # złączenie tabel
        final_form_poza = pd.concat([final_poza, Thatistrue_poza], axis=1)

    display(final_form_poza)
    print_min_max(final_form_poza['How much [%]'])


def Model_odtwarzanie(My_model, plik_csv):
    final = pd.DataFrame()
    Thatistrue = pd.DataFrame()
    for file in plik_csv['Name'].unique():
        One_person = plik_csv.loc[plik_csv['Name'] == file].drop('Name', axis=1)
        One_person = scaler_standard.transform(One_person)

        Prediction = My_model.predict(One_person)  # tutaj zamieniamy knn na inny model do predykcji

        # Obliczanie na ile osoba sprawdzana jest podobna do samej siebie
        df11 = pd.DataFrame(Prediction, columns=['Ho'])
        df_True = df11[df11["Ho"] == file]
        probability = df_True.size / df11.size * 100
        df = pd.DataFrame({'Name': file, 'To himself [%]': probability}, index=[0])
        final = pd.concat([final, df]).reset_index(drop=True)

        # Obliczanie do kogo osoba sprawdzana jest najbardziej podobna
        hois = df11.value_counts().idxmax()[0]
        howmuch = df11.value_counts(normalize=True).max() * 100
        datahow = pd.DataFrame({'Similar to': hois, 'How much [%]': howmuch}, index=[0])
        Thatistrue = pd.concat([Thatistrue, datahow]).reset_index(drop=True)
        # złączenie tabel
        final_form = pd.concat([final, Thatistrue], axis=1)
        final_form['Is the same person'] = (final_form['Name'] == final_form['Similar to'])
        final_form['More than >80%'] = (final_form['How much [%]'] >= 80)

    display(final_form)
    print_min_max(final_form['To himself [%]'])


def Model_test(My_model, X_test_my, y_test_my, pred, plik_csv, scaler):
    print_results(calculate_metrics(y_test_my, pred))
    fig, ax = plt.subplots(figsize=(6, 6))
    ConfusionMatrixDisplay.from_estimator(My_model, X_test_my, y_test_my, ax=ax, xticks_rotation='vertical')
    plt.title('Confusion matrix')
    plt.show()
    final_test = pd.DataFrame()
    Thatistrue = pd.DataFrame()
    for file in plik_csv['Name'].unique():
        One_person = plik_csv.loc[plik_csv['Name'] == file].drop('Name', axis=1)
        One_person = scaler.transform(One_person)

        Prediction = My_model.predict(One_person)  # tutaj zamieniamy knn na inny model do predykcji

        # Obliczanie na ile osoba sprawdzana jest podobna do samej siebie
        df11 = pd.DataFrame(Prediction, columns=['Ho'])
        df_True = df11[df11["Ho"] == file]
        probability = df_True.size / df11.size * 100
        df = pd.DataFrame({'Name': file, 'To himself [%]': probability}, index=[0])
        final_test = pd.concat([final_test, df]).reset_index(drop=True)

        # Obliczanie do kogo osoba sprawdzana jest najbardziej podobna
        hois = df11.value_counts().idxmax()[0]
        howmuch = df11.value_counts(normalize=True).max() * 100
        datahow = pd.DataFrame({'Similar to': hois, 'How much [%]': howmuch}, index=[0])
        Thatistrue = pd.concat([Thatistrue, datahow]).reset_index(drop=True)
        # złączenie tabel
        final_form = pd.concat([final_test, Thatistrue], axis=1)

    display(final_form)
    print_min_max(final_form['To himself [%]'])


def obrobka_mfcc(path, save):
    final = pd.DataFrame()
    file_list = os.listdir(path)
    for file in file_list:
        (sr, y) = wavfile.read(path + file)
        y = y * (1 / np.max(np.abs(y)))
        clips = librosa.effects.split(y, top_db=20)
        wav_data = []
        for c in clips:
            data = y[c[0]: c[1]]
            wav_data.extend(data)
        sf.write('5s.wav', wav_data, sr)
        (sr, y) = wavfile.read('5s.wav')
        mfcc_feat = mfcc(y, sr, winlen=winlen, winstep=winstep, numcep=numcep, nfilt=nfilt, nfft=nfft, lowfreq=lowfreq,
                         preemph=preemph, ceplifter=ceplifter, appendEnergy=appendEnergy, winfunc=winfunc)
        df_mfcc_feat = pd.DataFrame(mfcc_feat)
        df_mfcc_feat['Name'] = file
        final = pd.concat([final, df_mfcc_feat])
    final.reset_index(drop=True)
    final.to_csv(save + '.csv', index=False)