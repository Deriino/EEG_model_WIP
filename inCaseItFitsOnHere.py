import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy

import scipy as sp
import scipy.io as sio

import os

import random

import cv2 as cv

device = None
if torch.cuda.is_available():
  device = torch.device('cuda')
else:
  print('.###.')
  print('#####')
  print('#####')
  print('#####')
  print('#####')
  print('#####')
  print('.###.')
  print('.###.')
  print('.###.')
  print('..#..')
  print('.....')
  print('.###.')
  print('#####')
  print('.###.')
  print('.....')
  print('Oh NO! Cuda is not available! D:')
  device = torch.device('cpu')


class LinearLayer(nn.Module):
  def __init__(o, inChs, outChs, normChs, activ, useBatchNorm=True):
    super(LinearLayer, o).__init__()

    o.lin = nn.Linear(inChs, outChs)
    o.norm = nn.BatchNorm1d(normChs)
    o.activ = activ
    o.useBatchNorm = useBatchNorm
  
  def forward(o, x):
    x = o.lin(x)
    if o.useBatchNorm:
      x = o.norm(x)
    x = o.activ(x)
    return x


class EEG_Model_2(nn.Module):
  # Ch -> number of inputs
  def __init__(o, inputs=128, fftChannels=64, comprInputs=32, comprFftChannels=32, gruSize=256, outSize=9):
    super(EEG_Model_2, o).__init__()

    o.act = nn.LeakyReLU(0.2)
    
    o.fcChannels1 = LinearLayer(fftChannels, fftChannels, inputs, o.act)
    # permute(0,2,1)
    o.fcChannels2 = LinearLayer(inputs, inputs, fftChannels, o.act)
    # pernute(0,2,1)
    o.fcChannels3 = LinearLayer(fftChannels, comprFftChannels, inputs, o.act)
    # permute(0,2,1)
    o.fcChannels4 = LinearLayer(inputs, comprInputs, comprFftChannels, o.act)
    # pernute(0,2,1)

    o.fftChannels = fftChannels

    o.comprInputs = comprInputs
    o.comprFftChannels = comprFftChannels
    o.flatSize = comprInputs * comprFftChannels

    o.gru = nn.GRU(o.flatSize, gruSize)
    
    o.decide = nn.Linear(gruSize, outSize)

  def forward(o, x, h=None):
    

    firstDim, secondDim = x.shape[0], x.shape[1]
    #print(firstDim, secondDim)
    x = x.reshape(-1, x.shape[2], x.shape[3])
    
    x = o.fcChannels1(x)
    x = x.permute(0,2,1)
    x = o.fcChannels2(x)
    x = x.permute(0,2,1)
    x = o.fcChannels3(x)
    x = x.permute(0,2,1)
    x = o.fcChannels4(x)
    
    x = x.unsqueeze(0)
    #print(x.shape)
    x = x.view(firstDim, secondDim, o.comprFftChannels, o.comprInputs)
    #print(x.shape)
    x = x.permute(1,0,3,2)
    # ffts[timeChunk][N][C][time]
    #print(x.shape)

    # put it through a GRU
    x = x.reshape(-1, x.shape[1], o.flatSize)
    #print(x.shape)
    if h is None:
      x, h_new = o.gru(x)
    else:
      x, h_new = o.gru(x, h)

    #print(x.shape)
    x = o.decide(x).permute(0,2,1)
    #print(x.shape)

    return x, h_new
  
  def convertX(o, ffts):
    return torch.from_numpy(ffts.astype(numpy.float32)).to(device)[:,:,:,0:o.fftChannels]

  def convertY(o, y):
    return torch.from_numpy(y).to(device).long().permute(1,0)

def chooseData2(dataset, time=128, skip=1, amt=10, N=90, nOff=0):
  space = len(dataset[0][0])
  xarr = []
  yarr = []
  totalRoom = time + skip * (amt-1)
  space = len(dataset[0][0])
  for nnn in range(nOff, N+nOff):
    n = nnn % 9
    while True:
      x = random.randint(0, space-totalRoom)
      
      xlist, ylist = [], []
      for off in range(0, skip*amt, skip):
        xlist.append(dataset[n][0:128, (x + off):(x + off+time)])
        ylist.append(n)
      
      xarr.append(xlist)
      yarr.append(ylist)
      break
  arrx, arry = numpy.array(xarr), numpy.array(yarr)
  return arrx, arry

def chooseData(dataset, a=5000, b=5000, time=128, skip=1, amt=10, N=90, nOff=0, which='a'):
  #print(len(dataset[0])-1)
  # split the data into chunks
  xarr = []
  yarr = []
  totalRoom = time + skip * (amt-1)
  space = len(dataset[0][0])
  for nnn in range(nOff, N+nOff):
    n = nnn % 9
    while True:
      x = random.randint(0, space-totalRoom)
      
      
      if x % (a+b) + totalRoom == (x+totalRoom-1) % (a+b):
        if which == 'a' and (x+totalRoom-1) % (a+b) >= a:
          continue
        elif which == 'b' and x % (a+b) <= a:
          continue

      xlist, ylist = [], []
      for off in range(0, skip*amt, skip):
        xlist.append(dataset[n][:, (x + off):(x + off+time)])
        ylist.append(n)
      xarr.append(xlist)
      yarr.append(ylist)
      break
  #print(len(xarr),len(xarr[0]),xarr[0][0].shape)
  return numpy.array(xarr), numpy.array(yarr)

def runFFT(data):
  haveDoneFft = numpy.fft.fft(data)
  #print(haveDoneFft.shape)
  haveDoneFft = numpy.real(haveDoneFft)
  #haveDoneFft = numpy.absolute(haveDoneFft)
  return haveDoneFft

# it returns a numpy array
def loadDataFromFile(url):
  mat = sio.loadmat(url)['EEG'][0][0][15]
  return mat

def loadEntireDataset(folder):
  dataset, labels = [], []
  for url in os.listdir(folder):
    url2 = '../../%s/%s' % (folder, url)
    mat = loadDataFromFile(url2)
    dataset.append(mat)
    labels.append(url2)
  return dataset, labels

def oneTrain(model, optim, dataset, labels, batchSize, batchNum, train=True):
  loss = nn.CrossEntropyLoss()
  attempt = 0
  while True:
    try:
      data, labels = chooseData2(dataset)
      break
    except ValueError:
      attempt += 1
      continue
  
  # data[N][timeChunk][C][time]
  ffts = runFFT(data)
  # ffts[N][timeChunk][C][time]
  x = model.convertX(ffts)
  y = model.convertY(labels)
  # x[N][timeChunk][C][time]
  y_pred = model(x)[0]
  optim.zero_grad()

  if train:
    # print(y_pred.shape, y.shape)
    modelLoss = loss(y_pred, y)
    
    print(batchNum, modelLoss.item())
    #predictions = torch.argmax(y_pred, 2)
    #print(batchNum, '#######', predictions)
    modelLoss.backward()
    
    optim.step()
  else:
    predictions = torch.argmax(y_pred, 1)
    print(batchNum, '#######', predictions)
    correct, total = 0, 0
    hmm = ((predictions - y) == 0).sum(1)
    print(hmm / 0.9)
    del predictions
    
  del x
  del y
  del y_pred

def train(model, optim, dataset, labels, batchSize, numBatches):
  for batchNum in range(numBatches):
    oneTrain(
      model,
      optim,
      dataset,
      labels,
      batchSize,
      batchNum
    )

def train2(model, optim, datasetUrls, batchSize, numBatches, switchDataInterval):
  whichCounter = 0

  lastDatasetLoaded = ''
  for batchNum in range(numBatches):
    if cv.waitKey(10) == 's':
      print('saving...')
      torch.save(model.state_dict(), 'quicksave.model')
      print('saved')
    
    if lastDatasetLoaded == '' or batchNum % switchDataInterval == 0:
      if whichCounter % len(datasetUrls) == 0:
        switchDataInterval += 1
        switchDataInterval = min(5, switchDataInterval)
      
      toLoad = datasetUrls[whichCounter % len(datasetUrls)]
      
      if lastDatasetLoaded != toLoad:
        dataset, labels = loadEntireDataset(toLoad)
        lastDatasetLoaded = toLoad
      whichCounter += 1
      
    
    if batchNum + 100 >= numBatches or (batchNum+1) % 100 == 0:
      toLoad = 'datasetBig/L10'
      if lastDatasetLoaded != toLoad:
        dataset, labels = loadEntireDataset(toLoad)
        lastDatasetLoaded = toLoad
      oneTrain(
        model,
        optim,
        dataset,
        labels,
        batchSize,
        batchNum,
        False
      )
    else:
      oneTrain(
        model,
        optim,
        dataset,
        labels,
        batchSize,
        batchNum,
        True
      )

'''
def test(model, datasetUrls):
  whichCounter = 0
  for url in datasetUrls:
    dataset, labels = loadEntireDataset(url)
    for i in range(0, len(dataset[0])-128):
      data = runFFT(data)
  for url in datasetUrls:
'''



model = EEG_Model_2().to(device)
optim = torch.optim.Adam(model.parameters(), lr=1e-3)

datasetUrls = [
  'datasetBig/L01',
  'datasetBig/L02',
  'datasetBig/L03',
  'datasetBig/L04',
  'datasetBig/L05',
  'datasetBig/L06',
  'datasetBig/L07',
  'datasetBig/L08',
  'datasetBig/L09',
]

# initial, just to see if it works

train2(
    model,
    optim,
    datasetUrls,
    batchSize=32,
    numBatches=1100,
    switchDataInterval=0
)

# final
torch.save(model.state_dict(), 'EEG_model_2.mAAAEEEHHH')

