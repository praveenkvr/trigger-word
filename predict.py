import pyaudio
import wave
import time
import numpy as np
import os
import struct
from tensorflow.keras.models import load_model
from scipy import signal
from scipy.io import wavfile
from playsound import playsound
from matplotlib import pyplot as plt

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 2
WAVE_OUTPUT_FILENAME = "temp.wav"
PREDICTION_THRESHOLD = 0.4
MAX_CONSECUTIVE = 50
CHIME_PATH = os.path.abspath(os.path.join(
    'data', 'chime.wav')).replace(" ", "%20")

model = load_model('trained.h5')


def capture_audio():

    try:
        audio = pyaudio.PyAudio()
        stream = audio.open(format=FORMAT, channels=CHANNELS,
                            rate=RATE, input=True, output=True,
                            frames_per_buffer=CHUNK)
        print('Streaming ...')
        frames = []
        for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)

        # stream.stop_stream()
        # stream.close()
        # audio.terminate()
        wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
        return True
    except Exception as e:
        return False


def get_spectogram(file_path="temp.wav"):
    sample_rate, samples = wavfile.read(file_path)

    dims = samples.ndim

    if dims == 2:
        samples = samples[:, 0]

    pxx, freqs, bins, im = plt.specgram(samples, Fs=sample_rate)
    return pxx


def predict():
    pxx = get_spectogram()
    pxx = np.swapaxes(pxx, 0, 1)
    pxx = np.expand_dims(pxx, 0)

    predicted = model.predict([pxx])
    Ty = predicted.shape[1]
    consecutive_timesteps = 0

    for i in range(Ty):
        consecutive_timesteps += 1
        if predicted[0, i, 0] > PREDICTION_THRESHOLD and consecutive_timesteps > MAX_CONSECUTIVE:

            playsound(CHIME_PATH)
            consecutive_timesteps = 0
            break


def main():

    while True:
        is_captured = capture_audio()
        if is_captured:
            predict()
            time.sleep(0.1)


if __name__ == '__main__':
    main()
