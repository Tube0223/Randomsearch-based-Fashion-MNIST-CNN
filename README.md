Randomsearch-based-Fashion-MNIST-CNN
======================================

This code was created to improve the performance of CNN using Random-search.

System version
* Python : 3.8.5  
* Conda : 4.9.2   
* Spyder : 4.1.5   
* Pytorch : 1.7.1   
* Numpy : 1.19.2   
* Optuna : 2.3.0

This code used Fashion MNIST data.

(This code don't have k-fold cross validation,yet. I will append k-fold cross validation)

This is result.
```
runfile('D:/deeplearning/랜덤서치 교차검증빼고 완료.py', wdir='D:/deeplearning')
[I 2021-01-26 13:19:49,030] A new study created in memory with name: no-name-55c2ee59-4f99-4108-8eac-8b2e8f520fc6
  0%|          | 0/5 [00:00<?, ?it/s]총 배치의 수 : 120
 20%|██        | 1/5 [01:05<04:20, 65.05s/it][Epoch:    1] cost = 0.630193651
 40%|████      | 2/5 [02:10<03:15, 65.14s/it][Epoch:    2] cost = 0.36574164
 60%|██████    | 3/5 [03:09<02:06, 63.24s/it][Epoch:    3] cost = 0.323824257
 80%|████████  | 4/5 [04:05<01:01, 61.06s/it][Epoch:    4] cost = 0.294223696
100%|██████████| 5/5 [05:08<00:00, 61.67s/it][Epoch:    5] cost = 0.273668915

C:\Users\wpdhk\anaconda3\lib\site-packages\torchvision\datasets\mnist.py:63: UserWarning: test_data has been renamed data
  warnings.warn("test_data has been renamed data")
C:\Users\wpdhk\anaconda3\lib\site-packages\torchvision\datasets\mnist.py:53: UserWarning: test_labels has been renamed targets
  warnings.warn("test_labels has been renamed targets")
[I 2021-01-26 13:25:06,003] Trial 0 finished with value: 0.8387558960933171 and parameters: {'learning_rate': 0.002373973614111898, 'training_epochs': 5, 'layer_number': 2}. Best is trial 0 with value: 0.8387558960933171.
Accuracy: 0.8198999762535095
정확도 트루 펄스 :  tensor([True, True, True,  ..., True, True, True])
정확도 텐서 :  tensor(0.8199)
정확도 실수값 :  0.8198999762535095
precision :  [0.641 0.991 0.616 0.722 0.954 0.984 0.523 0.815 0.974 0.979]
recall :  [0.82603093 0.84054283 0.89927007 0.9081761  0.57159976 0.89945155
 0.6018412  0.97956731 0.95960591 0.9014733 ]
precision 매크로 평균 :  0.8199
recall 매크로 평균 :  0.8387558960933171
  0%|          | 0/5 [00:00<?, ?it/s]총 배치의 수 : 120
 20%|██        | 1/5 [01:39<06:37, 99.39s/it][Epoch:    1] cost = 0.712941825
 40%|████      | 2/5 [03:06<04:47, 95.70s/it][Epoch:    2] cost = 0.381970704
 60%|██████    | 3/5 [04:33<03:06, 93.04s/it][Epoch:    3] cost = 0.324291945
 80%|████████  | 4/5 [06:03<01:32, 92.19s/it][Epoch:    4] cost = 0.295007616
100%|██████████| 5/5 [07:32<00:00, 90.50s/it][Epoch:    5] cost = 0.269199818

[I 2021-01-26 13:32:47,366] Trial 1 finished with value: 0.8311896655877916 and parameters: {'learning_rate': 0.00513874307448072, 'training_epochs': 5, 'layer_number': 5}. Best is trial 0 with value: 0.8387558960933171.
  0%|          | 0/4 [00:00<?, ?it/s]Accuracy: 0.8105999827384949
정확도 트루 펄스 :  tensor([ True,  True,  True,  ...,  True,  True, False])
정확도 텐서 :  tensor(0.8106)
정확도 실수값 :  0.8105999827384949
precision :  [0.751 0.98  0.557 0.778 0.909 0.897 0.398 0.87  0.99  0.976]
recall :  [0.84858757 0.92890995 0.84650456 0.86636971 0.55630355 0.96972973
 0.70818505 0.97424412 0.69230769 0.92075472]
precision 매크로 평균 :  0.8106
recall 매크로 평균 :  0.8311896655877916
총 배치의 수 : 120
 25%|██▌       | 1/4 [01:05<03:17, 65.73s/it][Epoch:    1] cost = 0.559608281
 50%|█████     | 2/4 [02:12<02:11, 65.92s/it][Epoch:    2] cost = 0.318111718
 75%|███████▌  | 3/4 [03:18<01:06, 66.17s/it][Epoch:    3] cost = 0.28127864
100%|██████████| 4/4 [04:24<00:00, 66.19s/it][Epoch:    4] cost = 0.253935337

[I 2021-01-26 13:37:17,052] Trial 2 finished with value: 0.8221069537924732 and parameters: {'learning_rate': 0.009342973429749053, 'training_epochs': 4, 'layer_number': 3}. Best is trial 0 with value: 0.8387558960933171.
  0%|          | 0/4 [00:00<?, ?it/s]Accuracy: 0.7976999878883362
정확도 트루 펄스 :  tensor([True, True, True,  ..., True, True, True])
정확도 텐서 :  tensor(0.7977)
정확도 실수값 :  0.7976999878883362
precision :  [0.612 0.995 0.534 0.662 0.936 0.966 0.589 0.731 0.991 0.961]
recall :  [0.86808511 0.79282869 0.87112561 0.94706724 0.6023166  0.84440559
 0.56309751 0.9851752  0.82998325 0.91698473]
precision 매크로 평균 :  0.7977000000000001
recall 매크로 평균 :  0.8221069537924732
총 배치의 수 : 120
 25%|██▌       | 1/4 [00:28<01:25, 28.48s/it][Epoch:    1] cost = 0.550772071
 50%|█████     | 2/4 [00:57<00:57, 28.55s/it][Epoch:    2] cost = 0.285309613
 75%|███████▌  | 3/4 [01:25<00:28, 28.62s/it][Epoch:    3] cost = 0.254225701
100%|██████████| 4/4 [01:54<00:00, 28.56s/it][Epoch:    4] cost = 0.231969461

[I 2021-01-26 13:39:13,630] Trial 3 finished with value: 0.8531470056592149 and parameters: {'learning_rate': 0.007916465997285804, 'training_epochs': 4, 'layer_number': 1}. Best is trial 3 with value: 0.8531470056592149.
  0%|          | 0/4 [00:00<?, ?it/s]Accuracy: 0.8141000270843506
정확도 트루 펄스 :  tensor([True, True, True,  ..., True, True, True])
정확도 텐서 :  tensor(0.8141)
정확도 실수값 :  0.8141000270843506
precision :  [0.659 0.983 0.434 0.765 0.967 0.977 0.565 0.851 0.978 0.962]
recall :  [0.90273973 0.93619048 0.93939394 0.90212264 0.49794027 0.89305302
 0.5978836  0.97591743 0.94401544 0.94221352]
precision 매크로 평균 :  0.8141
recall 매크로 평균 :  0.8531470056592149
총 배치의 수 : 120
 25%|██▌       | 1/4 [01:30<04:31, 90.39s/it][Epoch:    1] cost = 0.789616346
 50%|█████     | 2/4 [03:02<03:01, 90.98s/it][Epoch:    2] cost = 0.396761179
 75%|███████▌  | 3/4 [04:35<01:31, 91.62s/it][Epoch:    3] cost = 0.338770598
100%|██████████| 4/4 [06:09<00:00, 92.38s/it][Epoch:    4] cost = 0.297944397

[I 2021-01-26 13:45:30,257] Trial 4 finished with value: 0.7714662489231923 and parameters: {'learning_rate': 0.0018403593114630554, 'training_epochs': 4, 'layer_number': 5}. Best is trial 3 with value: 0.8531470056592149.
Accuracy: 0.7178999781608582
정확도 트루 펄스 :  tensor([True, True, True,  ..., True, True, True])
정확도 텐서 :  tensor(0.7179)
정확도 실수값 :  0.7178999781608582
precision :  [0.529 0.983 0.293 0.471 0.847 0.977 0.518 0.633 0.986 0.942]
recall :  [0.83835182 0.75848765 0.9669967  0.83807829 0.50326797 0.77478192
 0.5469905  0.97987616 0.60639606 0.90143541]
precision 매크로 평균 :  0.7179
recall 매크로 평균 :  0.7714662489231923
```

After reading Kaintels's advice, I pulled this code out of random search code.   
Then, the run time was reduced.

```
runfile('D:/deeplearning/랜덤서치 교차검증빼고 완료.py', wdir='D:/deeplearning')
[I 2021-01-30 16:02:47,722] A new study created in memory with name: no-name-7f9ea0c6-a677-4298-a8b9-18a1e6dba9ee
  0%|          | 0/5 [00:00<?, ?it/s]총 배치의 수 : 120
 20%|██        | 1/5 [00:53<03:33, 53.26s/it][Epoch:    1] cost = 0.630193651
 40%|████      | 2/5 [01:48<02:41, 53.74s/it][Epoch:    2] cost = 0.36574164
 60%|██████    | 3/5 [02:41<01:47, 53.62s/it][Epoch:    3] cost = 0.323824257
 80%|████████  | 4/5 [03:36<00:53, 53.92s/it][Epoch:    4] cost = 0.294223696
100%|██████████| 5/5 [04:32<00:00, 54.46s/it][Epoch:    5] cost = 0.273668915

C:\Users\wpdhk\anaconda3\lib\site-packages\torchvision\datasets\mnist.py:63: UserWarning: test_data has been renamed data
  warnings.warn("test_data has been renamed data")
C:\Users\wpdhk\anaconda3\lib\site-packages\torchvision\datasets\mnist.py:53: UserWarning: test_labels has been renamed targets
  warnings.warn("test_labels has been renamed targets")
[I 2021-01-30 16:07:24,370] Trial 0 finished with value: 0.8387558960933171 and parameters: {'learning_rate': 0.002373973614111898, 'training_epochs': 5, 'layer_number': 2}. Best is trial 0 with value: 0.8387558960933171.
  0%|          | 0/5 [00:00<?, ?it/s]Accuracy: 0.8198999762535095
정확도 트루 펄스 :  tensor([True, True, True,  ..., True, True, True])
정확도 텐서 :  tensor(0.8199)
정확도 실수값 :  0.8198999762535095
precision :  [0.641 0.991 0.616 0.722 0.954 0.984 0.523 0.815 0.974 0.979]
recall :  [0.82603093 0.84054283 0.89927007 0.9081761  0.57159976 0.89945155
 0.6018412  0.97956731 0.95960591 0.9014733 ]
precision 매크로 평균 :  0.8199
recall 매크로 평균 :  0.8387558960933171
총 배치의 수 : 120
 20%|██        | 1/5 [01:30<06:03, 90.90s/it][Epoch:    1] cost = 0.712941825
 40%|████      | 2/5 [03:00<04:31, 90.63s/it][Epoch:    2] cost = 0.381970704
 60%|██████    | 3/5 [04:31<03:01, 90.62s/it][Epoch:    3] cost = 0.324291945
 80%|████████  | 4/5 [06:02<01:30, 90.59s/it][Epoch:    4] cost = 0.295007616
100%|██████████| 5/5 [07:32<00:00, 90.47s/it][Epoch:    5] cost = 0.269199818

[I 2021-01-30 16:15:03,519] Trial 1 finished with value: 0.8311896655877916 and parameters: {'learning_rate': 0.00513874307448072, 'training_epochs': 5, 'layer_number': 5}. Best is trial 0 with value: 0.8387558960933171.
  0%|          | 0/4 [00:00<?, ?it/s]Accuracy: 0.8105999827384949
정확도 트루 펄스 :  tensor([ True,  True,  True,  ...,  True,  True, False])
정확도 텐서 :  tensor(0.8106)
정확도 실수값 :  0.8105999827384949
precision :  [0.751 0.98  0.557 0.778 0.909 0.897 0.398 0.87  0.99  0.976]
recall :  [0.84858757 0.92890995 0.84650456 0.86636971 0.55630355 0.96972973
 0.70818505 0.97424412 0.69230769 0.92075472]
precision 매크로 평균 :  0.8106
recall 매크로 평균 :  0.8311896655877916
총 배치의 수 : 120
 25%|██▌       | 1/4 [01:06<03:18, 66.08s/it][Epoch:    1] cost = 0.559608281
 50%|█████     | 2/4 [02:12<02:12, 66.04s/it][Epoch:    2] cost = 0.318111718
 75%|███████▌  | 3/4 [03:18<01:06, 66.29s/it][Epoch:    3] cost = 0.28127864
100%|██████████| 4/4 [04:25<00:00, 66.32s/it][Epoch:    4] cost = 0.253935337

[I 2021-01-30 16:19:33,854] Trial 2 finished with value: 0.8221069537924732 and parameters: {'learning_rate': 0.009342973429749053, 'training_epochs': 4, 'layer_number': 3}. Best is trial 0 with value: 0.8387558960933171.
  0%|          | 0/4 [00:00<?, ?it/s]Accuracy: 0.7976999878883362
정확도 트루 펄스 :  tensor([True, True, True,  ..., True, True, True])
정확도 텐서 :  tensor(0.7977)
정확도 실수값 :  0.7976999878883362
precision :  [0.612 0.995 0.534 0.662 0.936 0.966 0.589 0.731 0.991 0.961]
recall :  [0.86808511 0.79282869 0.87112561 0.94706724 0.6023166  0.84440559
 0.56309751 0.9851752  0.82998325 0.91698473]
precision 매크로 평균 :  0.7977000000000001
recall 매크로 평균 :  0.8221069537924732
총 배치의 수 : 120
 25%|██▌       | 1/4 [00:28<01:25, 28.41s/it][Epoch:    1] cost = 0.550772071
 50%|█████     | 2/4 [00:56<00:56, 28.41s/it][Epoch:    2] cost = 0.285309613
 75%|███████▌  | 3/4 [01:25<00:28, 28.51s/it][Epoch:    3] cost = 0.254225701
100%|██████████| 4/4 [01:54<00:00, 28.71s/it][Epoch:    4] cost = 0.231969461

[I 2021-01-30 16:21:30,716] Trial 3 finished with value: 0.8531470056592149 and parameters: {'learning_rate': 0.007916465997285804, 'training_epochs': 4, 'layer_number': 1}. Best is trial 3 with value: 0.8531470056592149.
  0%|          | 0/4 [00:00<?, ?it/s]Accuracy: 0.8141000270843506
정확도 트루 펄스 :  tensor([True, True, True,  ..., True, True, True])
정확도 텐서 :  tensor(0.8141)
정확도 실수값 :  0.8141000270843506
precision :  [0.659 0.983 0.434 0.765 0.967 0.977 0.565 0.851 0.978 0.962]
recall :  [0.90273973 0.93619048 0.93939394 0.90212264 0.49794027 0.89305302
 0.5978836  0.97591743 0.94401544 0.94221352]
precision 매크로 평균 :  0.8141
recall 매크로 평균 :  0.8531470056592149
총 배치의 수 : 120
[Epoch:    1] cost = 0.789616346
 50%|█████     | 2/4 [03:03<03:03, 91.77s/it][Epoch:    2] cost = 0.396761179
 75%|███████▌  | 3/4 [04:35<01:31, 91.81s/it][Epoch:    3] cost = 0.338770598
100%|██████████| 4/4 [06:08<00:00, 92.07s/it][Epoch:    4] cost = 0.297944397

[I 2021-01-30 16:27:45,880] Trial 4 finished with value: 0.7714662489231923 and parameters: {'learning_rate': 0.0018403593114630554, 'training_epochs': 4, 'layer_number': 5}. Best is trial 3 with value: 0.8531470056592149.
Accuracy: 0.7178999781608582
정확도 트루 펄스 :  tensor([True, True, True,  ..., True, True, True])
정확도 텐서 :  tensor(0.7179)
정확도 실수값 :  0.7178999781608582
precision :  [0.529 0.983 0.293 0.471 0.847 0.977 0.518 0.633 0.986 0.942]
recall :  [0.83835182 0.75848765 0.9669967  0.83807829 0.50326797 0.77478192
 0.5469905  0.97987616 0.60639606 0.90143541]
precision 매크로 평균 :  0.7179
recall 매크로 평균 :  0.7714662489231923
```
