reference github : https://github.com/XiaoYee/emotion_classification


# Resnet_GRU

train command:
python main.py --optim adam --multi_gpu --length 4 --iter 1000

--resume : resume training, continue with latest checkpoint
--resume_best : resume training, continue with best checkpoint


if you want to debug sth, reduce iter(num of iters for each epoch) and check the bug



## 바꿔야 할 것
1. checkpoint path 조정 --> dir만들어서 분리하기
2. cuda available 확인해서 device 조정 ok
3. multi gpu도 자동 설정되게 변경 ok
4. training tqdm으로 바꾸기 ok
5. fine tuning
    1. length 바꾸기
    2. adam vs sgd
    3. attention 추가해볼까
    4. lr scheduler
    5. 이미지에 노이즈 추가
    6. trainig img histogram equalization?
6. face detection도 딥러닝으로 해서 정확도 높이기