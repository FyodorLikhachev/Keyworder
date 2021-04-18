import pyaudio
import time
import torchaudio
import threading
import torch
from torch import nn
import numpy as np

# region Model
# TODO: move to another place
class GRUModel(nn.Module):
    def __init__(self, input_size=40, hidden_size=256, num_classes=2):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=2)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.initHidden()
        self.inp_size = input_size

    def forward(self, x):
        x = x.view(-1, 1, self.inp_size)
        out, self.h1 = self.gru(x, self.h1)
        self.h1 = self.h1.detach()
        out = nn.Tanh()(out)
        out = self.fc(out)
        B, N, D = out.size()
        return out.view(-1, D)

    def initHidden(self):
        self.h1 = None
        self.h2 = None
# endregion

# list input devices
# TODO: move to another place
def list_input_devices():
    p = pyaudio.PyAudio()
    info = p.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')
    for i in range(0, numdevices):
        if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
            print("Input Device id ", i, " - ", p.get_device_info_by_host_api_device_index(0, i).get('name'))

list_input_devices()

RATE = 16000
WINDOW_TIME = 0.01
WINDOW_LENGTH = int(RATE * WINDOW_TIME)
CHUNK_SIZE = 1024
MAX_QUEUE_LENGTH = 5 # RATE // CHUNK_SIZE
MIN_VOLUME = 1000
MODEL_FILE = "/Users/user/Downloads/model.0.93.97.pt"  # TODO: move to args
SENSITIVITY = 20  # TODO: move to args
list = list()

def main():
    stopped = threading.Event()
    listen_t = threading.Thread(target=listen, args=(stopped, list,))

    try:
        listen_t.start()
        Keyworder(MODEL_FILE, list)
        listen_t.join()
    except KeyboardInterrupt:
        stopped.set()

class Keyworder:
    def __init__(self, model_file, list):
        self.model = torch.load(model_file, map_location='cpu')
        self.model.eval().to('cpu')
        self.list = list
        self.success_num = 0

        self.dispatch()  # starts queue dispatching

    def predict(self):
        waveform = torch.Tensor([a for a in self.list]).flatten()

        with torch.no_grad():
            mel = self.transform(waveform)
            pred = self.get_pred(mel)
            if pred == 1:
                self.success_num = self.success_num + 1
                print(f"Success #{self.success_num}")
                time.sleep(0.1)  # after key-word detection skip 100 ms

    def get_pred(self, x):
        preds = self.model(x).argmax(dim=1)
        detect_in_row = 0
        for y in preds:
            if y == 1:
                detect_in_row = detect_in_row + 1

                if detect_in_row >= SENSITIVITY:
                    return 1

        return 0

    def transform(self, audio):
        transform = torchaudio.transforms.MelSpectrogram(
            n_fft=WINDOW_LENGTH,
            win_length=WINDOW_LENGTH,
            hop_length=WINDOW_LENGTH // 2,
            power=1)

        return transform(audio).squeeze(0).t() # removes batch size

    def dispatch(self):
        while True:
            if len(self.list) > MAX_QUEUE_LENGTH:
                diff = len(self.list) - MAX_QUEUE_LENGTH
                for _ in range(diff):
                    self.list.pop(0)

                self.predict()
            elif len(self.list) == MAX_QUEUE_LENGTH:
                self.predict()
            else:
                time.sleep(0.05)  # waits 50 ms

def listen(stopped, list):
    stream = pyaudio.PyAudio().open(
        format=pyaudio.paFloat32,
        channels=1,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK_SIZE,
        input_device_index=0  # By default use Built-in Microphone
    )

    while True:
        if stopped.wait(timeout=0):
            break

        chunk = np.frombuffer(stream.read(CHUNK_SIZE), np.float32)
        list.append(chunk)

if __name__ == '__main__':
    main()
