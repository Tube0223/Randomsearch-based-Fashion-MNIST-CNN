import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset
import torch.nn.init

from tqdm import tqdm

import numpy as np
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

import optuna
from optuna.samplers import RandomSampler
import random
import numpy as np


random.seed(777)
np.random.seed(777)

device = 'cpu'
#device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 랜덤 시드 고정
torch.manual_seed(777)

# GPU 사용 가능일 경우 랜덤 시드 고정
#if device == 'cpu':
torch.cuda.manual_seed_all(777)
    
#tmp_data = torch.zeros(500, 1, 28, 28).cuda()
    
#tmp_label = torch.zeros(500, 1).cuda()

mnist_train = dsets.FashionMNIST(root='./data', # 다운로드 경로 지정
                             train=True, # True를 지정하면 훈련 데이터로 다운로드
                             transform=transforms.ToTensor(), # 텐서로 변환
                             download=True)

mnist_test = dsets.FashionMNIST(root='./data', # 다운로드 경로 지정
                            train=False, # False를 지정하면 테스트 데이터로 다운로드
                            transform=transforms.ToTensor(), # 텐서로 변환
                            download=True)

#tmp_datas = TensorDataset(mnist_train.data[:500].unsqueeze(1).float(), mnist_train.targets[:500]) # 소음 데이터로 나중에 바꾸기

def objective(trial):
  learning_rate = trial.suggest_uniform("learning_rate", 0.001, 0.01)
  training_epochs = trial.suggest_int("training_epochs", 2, 5)
  batch_size = 500
  layer_number = trial.suggest_int("layer_number", 1,5)
    
  data_loader = torch.utils.data.DataLoader(dataset=mnist_train,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            drop_last=True)

  class CNN(torch.nn.Module):

      def __init__(self):
          super(CNN, self).__init__()
        # 첫번째층
          self.layer1 = torch.nn.Sequential(
              torch.nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
              torch.nn.ReLU(),
              torch.nn.MaxPool2d(kernel_size=1, stride=1))

        # 두번째층
          self.layer2 = torch.nn.Sequential(
              torch.nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
              torch.nn.ReLU(),
              torch.nn.MaxPool2d(kernel_size=2, stride=2))
          
        # 세번째층
          self.layer3 = torch.nn.Sequential(
              torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
              torch.nn.ReLU(),
              torch.nn.MaxPool2d(kernel_size=1, stride=1))
          
        # 네번째층
          self.layer4 = torch.nn.Sequential(
              torch.nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
              torch.nn.ReLU(),
              torch.nn.MaxPool2d(kernel_size=2, stride=2))
          
          # 네번째층
          self.layer5 = torch.nn.Sequential(
              torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
              torch.nn.ReLU(),
              torch.nn.MaxPool2d(kernel_size=1, stride=1))

        # 전결합층 7x7x64 inputs -> 10 outputs
          self.fc1 = torch.nn.Linear(28 * 28 * 16, 10, bias=True)
          self.fc2 = torch.nn.Linear(14 * 14 * 16, 10, bias=True)
          self.fc3 = torch.nn.Linear(14 * 14 * 32, 10, bias=True)
          self.fc4 = torch.nn.Linear(7 * 7 * 32, 10, bias=True)
          self.fc5 = torch.nn.Linear(7 * 7 * 64, 10, bias=True)

        # 전결합층 한정으로 가중치 초기화
          torch.nn.init.xavier_uniform_(self.fc1.weight)
          torch.nn.init.xavier_uniform_(self.fc2.weight)
          torch.nn.init.xavier_uniform_(self.fc3.weight)
          torch.nn.init.xavier_uniform_(self.fc4.weight)
          torch.nn.init.xavier_uniform_(self.fc5.weight)



      def forward(self, x):
          if layer_number == 1:
              out = self.layer1(x)
              out = out.view(out.size(0), -1)   # 전결합층을 위해서 Flatten
              out = self.fc1(out)
          elif layer_number == 2:
              out = self.layer1(x)
              out = self.layer2(out)
              out = out.view(out.size(0), -1)   # 전결합층을 위해서 Flatten
              out = self.fc2(out)
          elif layer_number == 3:
              out = self.layer1(x)
              out = self.layer2(out)
              out = self.layer3(out)
              out = out.view(out.size(0), -1)   # 전결합층을 위해서 Flatten
              out = self.fc3(out)
          elif layer_number == 4:
              out = self.layer1(x)
              out = self.layer2(out)
              out = self.layer3(out)
              out = self.layer4(out)
              out = out.view(out.size(0), -1)   # 전결합층을 위해서 Flatten
              out = self.fc4(out)
          else:
              out = self.layer1(x)
              out = self.layer2(out)
              out = self.layer3(out)
              out = self.layer4(out)
              out = self.layer5(out)
              out = out.view(out.size(0), -1)   # 전결합층을 위해서 Flatten
              out = self.fc5(out)
          return out
    
  model = CNN().to(device)
  #torch.save(model.parameters(), "model.pth")

  criterion = torch.nn.CrossEntropyLoss().to(device)    # 비용 함수에 소프트맥스 함수 포함되어져 있음.
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

  total_batch = len(data_loader)
  print('총 배치의 수 : {}'.format(total_batch))

  total_acc = [] # 60000개  데이터를 교차검증을 12500개 test, 47500개 train으로 하나씩 정확도 append

  for epoch in tqdm(range(training_epochs)):
      avg_cost = 0

      for X, Y in data_loader: # 미니 배치 단위로 꺼내온다. X는 미니 배치, Y는 레이블.
        # image is already size of (28x28), no reshape
        # label is not one-hot encoded
          X = X.to(device)
          Y = Y.to(device)

          optimizer.zero_grad()
          hypothesis = model(X)
          cost = criterion(hypothesis, Y)
          cost.backward()
          optimizer.step()

          avg_cost += cost / total_batch

      print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, avg_cost))
    

      
# 학습을 진행하지 않을 것이므로 torch.no_grad()
  with torch.no_grad():
      true_number = 0
      total_number = 0
      X_test = mnist_test.test_data.view(len(mnist_test), 1, 28, 28).float().to(device)
      
      #X_test = X_test[0:3]
      Y_test = mnist_test.test_labels.to(device)
      #Y_test = Y_test[0:3]
      prediction = model(X_test)
      correct_prediction = torch.argmax(prediction, 1)
      correct_prediction_acc = torch.argmax(prediction, 1) == Y_test
      accuracy = correct_prediction_acc.float().mean()
      #print('f1_score:', f1_score)
      print('Accuracy:', accuracy.item())
      print('정확도 트루 펄스 : ', correct_prediction_acc)
      print('정확도 텐서 : ', accuracy)
      print('정확도 실수값 : ', accuracy.item())
      print('precision : ', precision_score(correct_prediction, Y_test, average = None))
      print('recall : ', recall_score(correct_prediction, Y_test, average = None))
      print('precision 매크로 평균 : ', precision_score(correct_prediction, Y_test, average = 'macro'))
      print('recall 매크로 평균 : ', recall_score(correct_prediction, Y_test, average = 'macro'))
      #print('정확도 평균 : ', true_number/total_number)
  #return f1_score(correct_prediction, Y_test, average = 'macro')
  #return accuracy.item() #정확도 출력
  return recall_score(correct_prediction, Y_test, average = 'macro')

study = optuna.create_study(direction = 'maximize', sampler=RandomSampler(seed=777))
study.optimize(objective, n_trials=5)




#print(precision_score(correct_prediction, Y_test, average = ))
#print(recall_score(correct_prediction, Y_test, average = ))
#print(f1_score(correct_prediction, Y_test, average = ))
