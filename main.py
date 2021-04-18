import pyaudio
import time
import threading
import torch
import numpy as np
import constants
from grumodel import GRUModel
from functions import transform
from train import train

list = list() # shared list between 2 threads

def main():
    stopped = threading.Event()
    listen_t = threading.Thread(target=listen, args=(stopped, list,))

    try:
        listen_t.start()
        Keyworder(constants.MODEL_FILE, list)
        listen_t.join()
    except KeyboardInterrupt:
        stopped.set()

# Live keyword detector
class Keyworder:
    def __init__(self, model_file, list):
        self.model = torch.load(model_file, map_location=constants.DEVICE)
        self.model.eval().to(constants.DEVICE)
        self.list = list
        self.success_num = 0

        self.dispatch()  # starts list dispatching

    def predict(self):
        # transform array values to 1-vector Tensor
        waveform = torch.Tensor([a for a in self.list]).flatten()

        with torch.no_grad():
            mel = transform(waveform).squeeze(0).t()
            pred = self.get_pred(mel)
            if pred == 1:
                self.success_num = self.success_num + 1
                print(f"Success #{self.success_num}")
                time.sleep(0.1)  # after keyword detection skip 100 ms

    # gets model prediction
    # if enough ones in a row is spotted then keyword has been detected
    def get_pred(self, x):
        preds = self.model(x).argmax(dim=1)
        detect_in_row = 0
        for y in preds:
            if y == 1:
                detect_in_row = detect_in_row + 1

                if detect_in_row >= constants.SENSITIVITY:
                    return 1

        return 0

    # main process for keyword prediction
    # if list exceeds MAX_QUEUE_LENGTH then deletes old values (circrular buffer)
    def dispatch(self):
        while True:
            if len(self.list) > constants.MAX_QUEUE_LENGTH:
                diff = len(self.list) - constants.MAX_QUEUE_LENGTH
                for _ in range(diff):
                    self.list.pop(0)

                self.predict()
            elif len(self.list) == constants.MAX_QUEUE_LENGTH:
                self.predict()
            else:
                time.sleep(0.05)  # waits 50 ms

# Adds audio stream to shared list by 64 ms
def listen(stopped, list):
    stream = pyaudio.PyAudio().open(
        format=pyaudio.paFloat32,
        channels=constants.CHANNEL_NUMBER,
        rate=constants.RATE,
        input=True,
        frames_per_buffer=constants.CHUNK_SIZE,
        input_device_index=constants.INPUT_DEVICE_INDEX # By default use Built-in Microphone
    )

    while True:
        if stopped.wait(timeout=0):
            break

        chunk = np.frombuffer(stream.read(constants.CHUNK_SIZE), np.float32)
        list.append(chunk)


if __name__ == '__main__':
    main()
    #model = train() # train for train =)
