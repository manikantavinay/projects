from flask import Flask, render_template ,redirect ,url_for;
import speech_recognition as sr
from flask import Flask, render_template ,redirect ,url_for;
from flask_wtf import FlaskForm;
from wtforms import FileField, SubmitField;
from werkzeug.utils import secure_filename;
import os;
from wtforms.validators import InputRequired;
import os;
os.environ['TF_ENABLE_ONEDNN_OPTS']='0';
os.environ['SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL']='True';
import sklearn.preprocessing;
from sklearn.preprocessing import StandardScaler;
scaler = StandardScaler();
from tensorflow.keras.models import load_model

import numpy as np;
import librosa;
import pandas as pd;
import numpy as np;
import os;
import seaborn as sns;
import matplotlib.pyplot as plt;
from sklearn.preprocessing import OneHotEncoder;
encoder = OneHotEncoder();

def extract_features(data):
    y=data
    #y,sr=librosa.load(filename,duration=3,offset=0.5)

    # ZCR
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=y).T, axis=0)
    result=np.hstack((result, zcr)) # stacking horizontally

    # Chroma_stft
    stft = np.abs(librosa.stft(data)) # Remove the attempt to unpack sample_rate here
    sample_rate = 44100
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft)) # stacking horizontally

    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc)) # stacking horizontally

    # Root Mean Square Value
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms)) # stacking horizontally

    # MelSpectogram
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel)) # stacking horizontally

    return result

def add_noise(data):
    """
    Add random noise to the audio data.
    """
    noise = np.random.randn(len(data))
    data_noise = data + 0.005 * noise
    return data_noise

def stretch(data, rate=1):
    """Stretches the audio data by a given rate."""
    return librosa.effects.time_stretch(data, rate=rate)

def pitch(data, sample_rate, n_steps=2):
    """Pitches the audio data by a given number of steps."""
    return librosa.effects.pitch_shift(data, sr=sample_rate, n_steps=n_steps)

def get_features(path):
    # duration and offset are used to take care of the no audio in start and the ending of each audio files as seen above.
    data, sample_rate = librosa.load(path, duration=2.5, offset=0.6)

    # without augmentation
    res1 = extract_features(data)
    result = np.array(res1)

    # data with noise
    noise_data =add_noise(data)
    res2 = extract_features(noise_data)
    result = np.vstack((result, res2)) # stacking vertically

    # data with stretching and pitching
    new_data = stretch(data)
    data_stretch_pitch = pitch(new_data, sample_rate)
    res3 = extract_features(data_stretch_pitch)
    result = np.vstack((result, res3)) # stacking vertically

    return result

import pandas as pd
Features = pd.read_csv(r'C:\Users\Dell\Desktop\JAVA\mini project\mini project\features.csv') # Read the entire DataFrame
X = Features.iloc[: ,:-1].values # Select all rows and all columns except the last one
Y = Features['labels'].values # Select the 'labels' column
print(Y)
print(Y.shape)



from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
Y = encoder.fit_transform(np.array(Y).reshape(-1,1)).toarray()

print("Encoded feature categories:", encoder.categories_)

print(Y.shape)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=0, shuffle=True)
x1=np.array(x_train)
x2=np.array(x_test)
y1=np.array(y_train)
y2=np.array(y_test)
x1.shape, y1.shape, x2.shape, y2.shape

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
x1=np.array(x_train)
x2=np.array(x_test)
x1.shape, y1.shape, x2.shape, y2.shape

x_train = np.expand_dims(x_train, axis=2)
x_test = np.expand_dims(x_test, axis=2)
x1=np.array(x_train)
x2=np.array(x_test)
x1.shape, y1.shape, x2.shape, y2.shape

#loaded_model=load_model('/content/drive/MyDrive/my_model.h5')

print(Y.shape)
loaded_model=load_model(r'C:\Users\Dell\Desktop\JAVA\mini project\mini project\my_model.h5')
def main1():
	r=sr.Recognizer()
	with sr.Microphone() as source:
		r.adjust_for_ambient_noise(source)
		audio=r.listen(source)
		print("recognizing")
		try:
			print("you"+r.recognize_google(audio))
			print("Audio Recorded ")
		except Exception as e:
			print("error "+str(e))
		with open("recorded.wav","wb") as f:   
			f.write(audio.get_wav_data())
			
app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'		
@app.route('/')
def index():
    return render_template('frontend.html')

@app.route('/record')
def record():
    return render_template('sample.html')
@app.route('/result',methods=['POST'])
def result():
    main1()
    features1=get_features(r'C:\Users\Dell\Desktop\JAVA\mini project\mini project\recorded.wav') # Then save the file
    features1 = scaler.transform(features1)
    features1 = np.expand_dims(features1, axis=2)
    pred_test =loaded_model.predict(features1,verbose=2)  # Predict directly on the features
    y_pred = encoder.inverse_transform(pred_test)
    a=y_pred.flatten()[0]
    return redirect(url_for("man", usr=a)) 
@app.route("/<usr>")
def man(usr):
    return render_template('home.html',data=usr)
if __name__ == '__main__':
    app.run(debug=True)