sample_rate = 16000
window_time = 0.01
hidden_size = 128
random_dataset_size = 1000

import torch
import torchaudio
import os
import random
from torch import nn
from jupyterplot import ProgressPlot

neg_clean_root = './out/neg_clean'
neg_noisy_root = './out/neg_noisy'
pos_clean_root = './out/pos_clean'
pos_noisy_root = './out/pos_noisy'
neg_rand_root =  './out/neg_random'

alina_root = './datasets/alina'
not_alina_root = './datasets/not_alina'
common_root = './datasets/common_dataset'

def load_audios(path):
  audios = []
  for filename in os.listdir(path):
    channels, rate = torchaudio.load(os.path.join(path, filename))
    resampler = torchaudio.transforms.Resample(rate, sample_rate)
    sample = resampler(channels[0])
    audios.append(sample)
  return audios

def load_random_dataset(path, dataset_size=1000, min_audio_size = 400, max_audio_size = 50000):
  dataset = []
  audios = load_audios(path)

  for i in range(dataset_size):
    audio = random.choice(audios)
    sample_size = random.randint(min_audio_size, max_audio_size)
    start_position =random.randint(0, len(audio)-sample_size)
    res_sample = audio[start_position:start_position+sample_size]
    dataset.append(res_sample)
  return dataset

neg_clean_dataset = load_audios(neg_clean_root)
neg_noisy_dataset = load_audios(neg_noisy_root)
pos_clean_dataset = load_audios(pos_clean_root)
pos_noisy_dataset = load_audios(pos_noisy_root)
neg_rand_dataset = load_random_dataset(neg_rand_root, dataset_size=2000, min_audio_size=4000, max_audio_size=50000)

class Dataset(torch.utils.data.Dataset):
  def to_dataset(self, dataset, is_alina):
    window_length = int(sample_rate*window_time)
    transform = torchaudio.transforms.MelSpectrogram(
      n_fft = window_length, 
      win_length = window_length,
      hop_length = window_length//2,
      power=1)
    dataset = [transform(data).t() for data in dataset]
    dataset =  [(data, torch.zeros(len(data), dtype = torch.long), 1 if is_alina else 0) for data in dataset]
    if is_alina:
      for data in dataset:
        n=len(data[1])
        for i in range(n):
          if i>=(n-23) and i<(n-0):
            data[1][i]=1
    return dataset

  def __init__(self, not_alina, alina):
    self.dataset = self.to_dataset(not_alina, False) + self.to_dataset(alina, True)

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    return self.dataset[idx]

dataset = Dataset(neg_clean_dataset+neg_noisy_dataset+neg_rand_dataset, pos_clean_dataset+pos_noisy_dataset)
clear_dataset = Dataset([], pos_clean_dataset)
 
train_size = int(len(dataset)/100*90)
test_size = len(dataset)-train_size
 
train_data, test_data = torch.utils.data.random_split(dataset, [train_size, test_size])

import matplotlib.pyplot as plt

plt.imshow(dataset[6000][0].t()., cmap='plasma')

class GRUModel(nn.Module):
  def __init__(self, input_size=40, hidden_size=256, num_classes = 2):
    super().__init__()
 
    self.gru = nn.GRU(input_size, hidden_size, num_layers=2)
    self.fc = nn.Linear(hidden_size, num_classes)
    self.initHidden()
    self.inp_size = input_size
 
  def forward(self, x):
    x = x.view(-1, 1, self.inp_size)
    out, self.h1 = self.rnn(x, self.h1)
    self.h1 = (self.h1[0].detach(), self.h1[1].detach())
    out = nn.Tanh()(out)
    out = self.fc(out)
    B, N, D = out.size()
 
    return out.view(-1, D)
 
  def initHidden(self):
    self.h1 = None
    self.h2 = None

def get_pred(model, x):
  cl = 1 if model(x.to(device))[-30:-1].argmax(dim=1).sum() > 0 else 0 
  return cl

from sklearn import metrics
 
def showMetrics(dataset, title):
  y_pred=[]
  y_true=[]
  for x, y, c in dataset:
    x = x.to(device)
    y = y.to(device)
    c = c
    pred = get_pred(model, x)

    y_pred.append(pred)
    y_true.append(c)
    
  alina_correct = [y_true == y_pred for y_true, y_pred in zip(y_true, y_pred) if y_true == 1]
  common_correct = [y_true == y_pred for y_true, y_pred in zip(y_true, y_pred) if y_true == 0]
 
  accuracy = metrics.accuracy_score(y_true, y_pred)
  precision = metrics.precision_score(y_true, y_pred)
  recall = metrics.recall_score(y_true, y_pred)
 
  print(title)
  print("Accuracy: {0} Precision: {1} Recall: {2}".format(round(accuracy, 2), round(precision,2), round(recall,2)))
  print("{0} из {1} Алина".format(sum(alina_correct),len(alina_correct)))
  print("{0} из {1} common".format(sum(common_correct), len(common_correct)))

def GetAccuracy(model, dataset):
  correct = 0
  for i, (x, y, c) in enumerate(dataset):
    pred = get_pred(model, x)
    if c == pred:
      correct+=1
  return correct/len(dataset)

# Train
device = torch.device(0)
input_size = 128

model = GRUModel(input_size=input_size, hidden_size=hidden_size, num_classes=2).to(device)
 
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
criterion = nn.CrossEntropyLoss()

pp = ProgressPlot(plot_names=['Accuracy', 'Loss'], line_names = ['Train', 'Test'])
 
loss_history = []
for epoch in range(100):
  for i, (x, y, c) in enumerate(train_data):
    optimizer.zero_grad()
    out = model(x.to(device))
    loss = criterion(out, y.to(device))
    loss.backward()
    optimizer.step()
    if i%5==0:
      model.initHidden()
    
    loss_history.append(loss)
    if len(loss_history) == 2000:
      loss_sum = sum(loss_history).item()
      acc_data = torch.utils.data.Subset(train_data, range(0, 700))
      pp.update([[GetAccuracy(model, acc_data), GetAccuracy(model, test_data)], 
                 [loss_sum, loss_sum]])
      loss_history=[]
  
  showMetrics(test_data, f"\nEpoch: {epoch}")
  showMetrics(Dataset(neg_rand_dataset, pos_clean_dataset), f'\nEpoch {epoch} Clear')

# Save PyTorch model
torch.save(model, 'model.0.93.97.pt')
