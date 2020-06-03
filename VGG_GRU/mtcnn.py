!pip install /kaggle/input/facenet-pytorch-vggface2/facenet_pytorch-2.2.7-py3-none-any.whl

from facenet_pytorch import MTCNN
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from tqdm.notebook import tqdm

## Create face detector ##

# default로 image normalization되는데
# post_process=False: 이거 안하고 사람 눈에 정상적으로 보이는 이미지를 원할 때
# margin = 20: 얼굴 주변에 margin    .detect에서는 안쓰임 원하면 쓰기
# select_largest=False: 얼굴이 size 말고 detection probability따라 정렬
# keep_all=True: multiple faces in a single image
mtcnn = MTCNN(select_largest=False, keep_all=True, post_process=False, device='cuda')


## Detect faces in batch ##

batch_size = 4
frames = []
faces = []
# faces = mtcnn(frames)

# Detection & bounding box랑 facial landmark return
boxes, probs, landmarks = mtcnn.detect(frame, landmarks=True)

# Visualize
fig, ax = plt.subplots(figsize=(16, 12))
ax.imshow(frame)
ax.axis('off')

for box, landmark in zip(boxes, landmarks):
    ax.scatter(*np.meshgrid(box[[0, 2]], box[[1, 3]]))
    ax.scatter(landmark[:, 0], landmark[:, 1], s=8)
fig.show()

# 혹시 저장할거면
# save_paths = [f'image_{i}.jpg' for i in range(len(frames))]
# mtcnn(frames, save_path=save_paths);

## Detection for real-time
'''
batch_size = 32
frames = []
boxes = []
landmarks = []
view_frames = []
view_boxes = []
view_landmarks = []

success, frame = cv2.VideoCapture.read()
if not success:
    continue

frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
frame = Image.fromarray(frame)
frame = frame.resize([int(f * 0.25) for f in frame.size])
frames.append(frame)

# When batch is full, detect faces and reset batch list
if len(frames) >= batch_size:
        batch_boxes, _, batch_landmarks = mtcnn.detect(frames, landmarks=True)
        boxes.extend(batch_boxes)
        landmarks.extend(batch_landmarks)
        
        view_frames.append(frames[-1])
        view_boxes.append(boxes[-1])
        view_landmarks.append(landmarks[-1])
        
        frames = []

# Visualize
fig, ax = plt.subplots(3, 3, figsize=(18, 12))
for i in range(9):
    ax[int(i / 3), i % 3].imshow(view_frames[i])
    ax[int(i / 3), i % 3].axis('off')
    for box, landmark in zip(view_boxes[i], view_landmarks[i]):
        ax[int(i / 3), i % 3].scatter(*np.meshgrid(box[[0, 2]], box[[1, 3]]), s=8)
        ax[int(i / 3), i % 3].scatter(landmark[:, 0], landmark[:, 1], s=6)
'''
