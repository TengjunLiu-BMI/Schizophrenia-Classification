import torch
import torch.nn as nn
import torch.utils.data as data
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import h5py
from sklearn.model_selection import KFold
import os
from torch.optim.lr_scheduler import StepLR
import scipy.io as io


# custom weights initialization
def weights_init(m):
    className = m.__class__.__name__
    if className.find('Conv') != -1:
        torch.nn.init.trunc_normal_(m.weight)
        torch.nn.init.zeros_(m.bias)
    elif className.find('Linear') != -1:
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)


def one_hot(labels,Label_class):

    one_hot_label = np.array([[int(i == int(labels[j])) for i in range(Label_class)] for j in range(len(labels))])

    return one_hot_label


# Regularized loss function
def regularized_loss(output, target, weights_fc1, weights_fc2, L2_penalty=0, L1_penalty=0):
    """loss function with L2 and L1 regularization
  Args:
    output (torch.Tensor): output of network
    target (torch.Tensor): neural response network is trying to predict
    weights (torch.Tensor): linear layer weights from neurons to hidden units (net.in_layer.weight)
    L2_penalty : scaling factor of sum of squared weights
    L1_penalty : scaling factor for sum of absolute weights
  Returns:
    (torch.Tensor) MSE error with L1 and L2 penalties added
  """

    loss_fn = nn.MSELoss()
    loss = loss_fn(output, target)

    L2 = L2_penalty * (torch.square(weights_fc1).sum() + torch.square(weights_fc2).sum())
    L1 = L1_penalty * (torch.abs(weights_fc1).sum() + torch.abs(weights_fc2).sum())
    loss += L1 + L2

    return loss


class SpecCNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SpecCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=hidden_size[0], kernel_size=3, stride=1, padding=1,
                      padding_mode='replicate'),
            nn.BatchNorm2d(hidden_size[0]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )  # output_size: input_size/2
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=hidden_size[1], kernel_size=3),
            nn.BatchNorm2d(hidden_size[1]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )  # output_size: (input_size/2-1)/2
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=hidden_size[2], kernel_size=2, padding=1,
                      padding_mode='replicate'),
            nn.BatchNorm2d(hidden_size[2]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )  # output_size: ((input_size/2-1)/2-1)/2
        self.drop = nn.Dropout(p=0.25)
        self.fc1 = nn.Linear(32 * 1 * 3, 10)
        self.fc2 = nn.Linear(10, output_size)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.drop(x)
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        out = F.softmax(self.fc2(x), dim=1)
        return out


def train(model, num_epochs=150):
    acc = []
    acc_ts = []
    loss2 = []
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.view(images.shape).requires_grad_().to(device)
            images = images.unsqueeze(1)
            labels = labels.float().to(device)
            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()
            # Forward pass to get output/logits
            outputs = model(images)
            # print(outputs)
            # Calculate Loss: softmax --> cross entropy loss
            loss = regularized_loss(outputs, labels, model.fc1.weight, model.fc2.weight, L1_penalty=L1Penalty)
            # Getting gradients w.r.t. parameters
            loss.backward()
            # Updating parameters
            optimizer.step()
        scheduler.step()
        accuracy = test(model, train_loader)
        ts_acc = test(model,test_loader)
        torch.save(model, '.\model_individual\model -epoch '+str(epoch)+'-k '+str(k) + ' -rand_seed ' + str(rand_seed) + '.pth')
        # if ts_acc > best_accuracy and accuracy > 90:
        #     torch.save(model_individual, '.\model_individual\model_' + str(ts_acc) + '-readout-1layer-v1.pth')
        #     best_accuracy = ts_acc
        acc.append(accuracy)
        acc_ts.append(ts_acc)
        loss2.append(loss.item())
        print('epoch: ', epoch, '. Loss: ', loss.item(), '. Tr Accuracy: ', accuracy, '. Ts Accuracy: ', ts_acc)
    return acc, acc_ts, loss2


def test(model, dataloader):
    correct = 0
    total = 0
    # Iterate through test dataset
    for images, labels in dataloader:
        images = images.view(images.shape).to(device)
        images = images.unsqueeze(1)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        if torch.cuda.is_available():
            correct += (predicted.cpu() == np.argmax(labels.long().cpu(), axis=1)).sum()
        else:
            correct += (predicted == labels).sum()

    accuracy = 100. * correct.numpy() / total
    return accuracy


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
plt.switch_backend("agg")

if not os.path.exists('.\model_individual'):
    os.makedirs('.\model_individual')
if not os.path.exists('.\matData'):
    os.makedirs('.\matData')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device)

# Define hyper-parameters
hiddenSize = [32, 32, 32]
outputSize = 2
L1Penalty = 0.01
LR = 0.001
beta1 = 0.9  # momentum in Adam, default
beta2 = 0.999  # momentum in Adam, default
batchSize = 16
numEpoch = 200
stepLrSize = 30
rand_seed = 57
numK = 2
torch.manual_seed(rand_seed)

# train on the dataset of Poland
fileNamePoland = 'dataset_Individual_Poland.mat'
Poland_mat = h5py.File(fileNamePoland, 'r')
dataPoland = Poland_mat['specFeas']
labelPoland = Poland_mat['labels']
dataPoland = np.transpose(dataPoland, (0, 2, 1))
labelPoland = np.transpose(labelPoland, (1, 0)).flatten()
labelPoland = one_hot(labelPoland, outputSize)

kf = KFold(n_splits=numK)
k = 0
for train_idx, val_idx in kf.split(dataPoland):
    k += 1
    # if k != kSet:
    #     continue
    train_X, train_y = dataPoland[train_idx], labelPoland[train_idx]
    val_X, val_y = dataPoland[val_idx], labelPoland[val_idx]
    print('training dataset shape: ', train_X.shape)
    print('training dataset(label) shape: ', train_y.shape)
    print('validating dataset shape: ', val_X.shape)
    tensor_trainX = torch.Tensor(train_X).to(device)  # transform to torch tensor
    tensor_trainY = torch.Tensor(train_y).to(device)
    train_dataset = data.TensorDataset(tensor_trainX, tensor_trainY)
    train_loader = data.DataLoader(train_dataset, batch_size=batchSize, shuffle=True)
    tensor_valX = torch.Tensor(val_X).to(device)  # transform to torch tensor
    tensor_valY = torch.Tensor(val_y).to(device)
    test_dataset = data.TensorDataset(tensor_valX, tensor_valY)
    test_loader = data.DataLoader(test_dataset, batch_size=batchSize, shuffle=False)

    dataShape = np.shape(train_X)
    inputSize = dataShape[1:2]

    model_CNN = SpecCNN(inputSize, hiddenSize, outputSize)
    model_CNN.to(device)

    optimizer = torch.optim.Adam(model_CNN.parameters(), lr=LR)
    scheduler = StepLR(optimizer, step_size=stepLrSize, gamma=0.5)

    acc, acc_val, loss = train(model_CNN, numEpoch)

    mat_path = '.\matData\SZ Classification Poland Individual k ' + str(k) + '_' + str(numK) + ' rand_seed ' + str(rand_seed) + '.mat'
    data2saveDic = {"acc": acc,
                    "acc_val": acc_val,
                    "loss": loss}
    io.savemat(mat_path, data2saveDic)

