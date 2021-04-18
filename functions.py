from torchaudio.transforms import MelSpectrogram
from constants import WINDOW_LENGTH, WINDOW_HOP

transform = MelSpectrogram(
    n_fft=WINDOW_LENGTH,
    win_length=WINDOW_LENGTH,
    hop_length=WINDOW_HOP,
    power=1
)
