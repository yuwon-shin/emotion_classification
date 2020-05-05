import os
import torch
from data_loader import FER
from torch.utils.data import DataLoader
from tqdm import tqdm
# from tensorboardX import SummaryWriter
import model as md



# train_writer = SummaryWriter(log_dir="log_last_last_last/train")
# valid_writer = SummaryWriter(log_dir="log_last_last_last/valid")

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
lr = 1e-6
epochs = 150
batch_size = 16

train_data_path = '../../../data/face_data'
train_dataset = FER(train_data_path , image_size=64, mode='train')
train_dataloader = DataLoader(train_dataset,  batch_size=batch_size, shuffle = True)

valid_data_path = '../../../data/face_data'
valid_dataset = FER(valid_data_path,image_size=64, mode='val')
valid_dataloader = DataLoader(valid_dataset,  batch_size=batch_size, shuffle = False)


# model = md.vgg16(pretrained = True, num_classes = 3).to(device)



model_name = 'vgg16'
feature_extract = True
num_classes = 3
model = md.init_pretrained_models(model_name, num_classes, feature_extract, use_pretrained=True)


model.to(device)
criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(params = model.parameters(), lr = lr)

...
for epoch in range(epochs):
    running_loss = 0
    running_acc = 0
    train_loss = 0
    model.train()

    # ================== Training ==================
    for image, label in tqdm(train_dataloader, desc="Epoch [%d/%d]" % (epoch + 1, epochs)):
        optimizer.zero_grad()  # Optimizer를 0으로 초기화
        image = image / 255.
        pred = model(image.float().transpose(3,2).transpose(2,1).to(device))
        loss = criterion(pred, label.to(device))

        loss.backward()
        optimizer.step()

        Softmax = torch.nn.Softmax(dim=1)
        _, prediction_tr = torch.max(Softmax(pred), 1)

        y_true_tr = label.cpu().detach().numpy()
        y_pred_tr = prediction_tr.cpu().detach().numpy()
        # acc = confusion_matrix(y_true, y_pred)

        acc_tr = ((label == prediction_tr.cpu()).sum().item() / pred.shape[0]) * 100


        # running_loss += loss.item()
        running_loss += loss * image.size(0)
        running_acc += acc_tr * image.size(0)

    train_loss = running_loss / len(train_dataset)
    train_acc = running_acc / len(train_dataset)

    # loss_sum = tf.summary.scalar("train_loss", train_loss)
    # acc_sum = tf.summary.scalar("train_accuracy", train_acc)


    # writer = tf.summary.FileWriter("./abc")

    # summary, _ = sess.run([loss_sum, epochs], feed_dict={x: loss_sum, y: epochs})


    print('>>> Train loss : %.4f - Train acc : %.4f'% (train_loss, train_acc))
    # train_acc = running_acc / len(train_dataloader)


    # =================== Validation ===================
    running_loss = 0
    running_acc = 0
    model.eval()
    # model.load_state_dict(torch.load('filenname'))



    with torch.no_grad():
        # val_st    ep = 0
        for image, label in valid_dataloader:
            image = image / 255.

            pred = model(image.float().transpose(3,2).transpose(1,2).to(device))
            loss = criterion(pred, label.to(device))


            Softmax = torch.nn.Softmax(dim=1)
            _, prediction = torch.max(Softmax(pred), 1)

            y_true = label.cpu().detach().numpy()
            y_pred = prediction.cpu().detach().numpy()
            # acc = confusion_matrix(y_true, y_pred)
            acc_tr = ((label == prediction.cpu()).sum().item() / pred.shape[0]) * 100
            # running_acc += acc_tr

            # running_loss += loss.item()
            # val_step +=1
            running_loss += loss.item() * image.size(0)
            running_acc += acc_tr * image.size(0)

    valid_loss = running_loss / len(valid_dataset)
    valid_acc = running_acc / len(valid_dataset)

    print(">>> Valid loss : %.4f - Valid acc : %.4f\n" % (valid_loss, valid_acc))
    print(prediction)
    print(label)
    print()

    # train_writer.add_scalar('loss', train_loss, epoch)
    # train_writer.add_scalar('accuracy', train_acc, epoch)
    # valid_writer.add_scalar('loss', valid_loss, epoch)
    # valid_writer.add_scalar('accuracy', valid_acc, epoch)

    if (epoch+1) % 5 == 0 :
        save_path = os.path.join('.', 'save_model')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model, os.path.join(save_path, 'model_epoch%04d_loss_%.4f_acc_%.4f.ckpt'%(epoch, valid_loss, valid_acc)))

