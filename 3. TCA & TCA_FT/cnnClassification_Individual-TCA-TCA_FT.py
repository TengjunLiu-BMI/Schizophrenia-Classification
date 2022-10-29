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
import scipy.linalg
import sklearn.metrics
from sklearn.neighbors import KNeighborsClassifier


def kernel(ker, X1, X2, gamma):
    K = None
    if not ker or ker == 'primal':
        K = X1
    elif ker == 'linear':
        if X2 is not None:
            K = sklearn.metrics.pairwise.linear_kernel(
                np.asarray(X1).T, np.asarray(X2).T)
        else:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1).T)
    elif ker == 'rbf':
        if X2 is not None:
            K = sklearn.metrics.pairwise.rbf_kernel(
                np.asarray(X1).T, np.asarray(X2).T, gamma)
        else:
            K = sklearn.metrics.pairwise.rbf_kernel(
                np.asarray(X1).T, None, gamma)
    return K


# custom weights initialization
def weights_init(m):
    className = m.__class__.__name__
    if className.find('Conv') != -1:
        torch.nn.init.trunc_normal_(m.weight)
        torch.nn.init.zeros_(m.bias)
    elif className.find('Linear') != -1:
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)


def one_hot(labels, Label_class):
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
            nn.Conv1d(in_channels=1, out_channels=hidden_size[0], kernel_size=3, stride=1, padding=1,
                      padding_mode='replicate'),
            nn.BatchNorm1d(hidden_size[0]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )  # output_size: input_size/2
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=hidden_size[1], kernel_size=3),
            nn.BatchNorm1d(hidden_size[1]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )  # output_size: (input_size/2-1)/2
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=hidden_size[2], kernel_size=2, padding=1,
                      padding_mode='replicate'),
            nn.BatchNorm1d(hidden_size[2]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )  # output_size: ((input_size/2-1)/2-1)/2
        self.drop = nn.Dropout(p=0.25)
        self.fc1 = nn.Linear(32 * 1, 10)
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


def train(model, train_loader, test_loader, num_epochs=150):
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
        ts_acc = test(model, test_loader)
        torch.save(model,
                   '.\model_TCA\model Individual -epoch ' + str(epoch) + '-k_P ' + str(k_Poland) + '_' + str(numK_Poland) +
                   '-k_R ' + str(k_Russia) + '_' + str(numK_Russia) + ' -rand_seed ' + str(rand_seed) + '.pth')
        # if ts_acc > best_accuracy and accuracy > 90:
        #     torch.save(model_Individual, '.\model_Individual\model_' + str(ts_acc) + '-readout-1layer-v1.pth')
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


class TCA:
    def __init__(self, kernel_type='primal', dim=30, lamb=1, gamma=1):
        '''
        Init func
        :param kernel_type: kernel, values: 'primal' | 'linear' | 'rbf'
        :param dim: dimension after transfer
        :param lamb: lambda value in equation
        :param gamma: kernel bandwidth for rbf kernel
        '''
        self.kernel_type = kernel_type
        self.dim = dim
        self.lamb = lamb
        self.gamma = gamma

    def fit(self, Xs, Xt):
        '''
        Transform Xs and Xt
        :param Xs: ns * n_feature, source feature
        :param Xt: nt * n_feature, target feature
        :return: Xs_new and Xt_new after TCA
        '''
        X = np.hstack((Xs.T, Xt.T))
        X /= np.linalg.norm(X, axis=0)
        m, n = X.shape
        ns, nt = len(Xs), len(Xt)
        e = np.vstack((1 / ns * np.ones((ns, 1)), -1 / nt * np.ones((nt, 1))))
        M = e * e.T
        M = M / np.linalg.norm(M, 'fro')
        H = np.eye(n) - 1 / n * np.ones((n, n))
        K = kernel(self.kernel_type, X, None, gamma=self.gamma)
        n_eye = m if self.kernel_type == 'primal' else n
        a, b = K @ M @ K.T + self.lamb * np.eye(n_eye), K @ H @ K.T
        w, V = scipy.linalg.eig(a, b)
        ind = np.argsort(w)
        A = V[:, ind[:self.dim]]
        Z = A.T @ K
        Z /= np.linalg.norm(Z, axis=0)

        Xs_new, Xt_new = Z[:, :ns].T, Z[:, ns:].T
        return Xs_new, Xt_new

    def fit_predict(self, Xs, Ys, Xt, Yt):
        '''
        Transform Xs and Xt, then make predictions on target using 1NN
        :param Xs: ns * n_feature, source feature
        :param Ys: ns * 1, source label
        :param Xt: nt * n_feature, target feature
        :param Yt: nt * 1, target label
        :return: Accuracy and predicted_labels on the target domain
        '''
        Xs_new, Xt_new = self.fit(Xs, Xt)
        clf = KNeighborsClassifier(n_neighbors=1)
        clf.fit(Xs_new, Ys.ravel())
        y_pred = clf.predict(Xt_new)
        acc = sklearn.metrics.accuracy_score(Yt, y_pred)

        return acc, y_pred

    # TCA code is done here. You can ignore fit_new and fit_predict_new.

    def fit_new(self, Xs, Xt, Xt2):
        '''
        Map Xt2 to the latent space created from Xt and Xs
        :param Xs : ns * n_feature, source feature
        :param Xt : nt * n_feature, target feature
        :param Xt2: n_s, n_feature, target feature to be mapped
        :return: Xt2_new, mapped Xt2 with projection created by Xs and Xt
        '''
        # Computing projection matrix A from Xs an Xt
        X = np.hstack((Xs.T, Xt.T))
        X /= np.linalg.norm(X, axis=0)
        m, n = X.shape
        ns, nt = len(Xs), len(Xt)
        e = np.vstack((1 / ns * np.ones((ns, 1)), -1 / nt * np.ones((nt, 1))))
        M = e * e.T
        M = M / np.linalg.norm(M, 'fro')
        H = np.eye(n) - 1 / n * np.ones((n, n))
        K = kernel(self.kernel_type, X, None, gamma=self.gamma)
        n_eye = m if self.kernel_type == 'primal' else n
        a, b = np.linalg.multi_dot(
            [K, M, K.T]) + self.lamb * np.eye(n_eye), np.linalg.multi_dot([K, H, K.T])
        w, V = scipy.linalg.eig(a, b)
        ind = np.argsort(w)
        A = V[:, ind[:self.dim]]

        Z = A.T @ K
        Z /= np.linalg.norm(Z, axis=0)

        Xs_new, Xt_new = Z[:, :ns].T, Z[:, ns:].T

        # Compute kernel with Xt2 as target and X as source
        Xt2 = Xt2.T
        K = kernel(self.kernel_type, X1=Xt2, X2=X, gamma=self.gamma)

        # New target features
        Xt2_new = K @ A

        return Xs_new, Xt_new, Xt2_new

    def fit_predict_new(self, Xt, Xs, Ys, Xt2, Yt2):
        '''
        Transfrom Xt and Xs, get Xs_new
        Transform Xt2 with projection matrix created by Xs and Xt, get Xt2_new
        Make predictions on Xt2_new using classifier trained on Xs_new
        :param Xt: ns * n_feature, target feature
        :param Xs: ns * n_feature, source feature
        :param Ys: ns * 1, source label
        :param Xt2: nt * n_feature, new target feature
        :param Yt2: nt * 1, new target label
        :return: Accuracy and predicted_labels on the target domain
        '''
        Xs_new, _ = self.fit(Xs, Xt)
        Xt2_new = self.fit_new(Xs, Xt, Xt2)
        clf = KNeighborsClassifier(n_neighbors=1)
        clf.fit(Xs_new, Ys.ravel())
        y_pred = clf.predict(Xt2_new)
        acc = sklearn.metrics.accuracy_score(Yt2, y_pred)

        return acc, y_pred


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
plt.switch_backend("agg")

if not os.path.exists('.\model_TCA'):
    os.makedirs('.\model_TCA')
if not os.path.exists('.\matData'):
    os.makedirs('.\matData')
if not os.path.exists('.\matTcaData'):
    os.makedirs('.\matTcaData')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device)

# Define hyper-parameters
hiddenSize = [32, 32, 32]
outputSize = 2
L1Penalty = 0.01
LR = 0.001
beta1 = 0.9  # momentum in Adam, default
beta2 = 0.999  # momentum in Adam, default
batchSize = 32
numEpoch = 100
stepLrSize = 30
rand_seed = 37
kSet = 1
numK_Russia = 10
numK_Poland = 2
torch.manual_seed(rand_seed)

# train on the dataset of Poland
fileNamePoland = 'dataset_Individual_Poland.mat'
Poland_mat = h5py.File(fileNamePoland, 'r')
dataPoland = Poland_mat['specFeas']
labelPoland = Poland_mat['labels']
dataPoland = np.transpose(dataPoland, (0, 2, 1))
labelPoland = np.transpose(labelPoland, (1, 0)).flatten()
labelPoland = one_hot(labelPoland, outputSize)

# test on the dataset of Russia
fileNameRussia = 'dataset_Individual_Russia.mat'
Russia_mat = h5py.File(fileNameRussia, 'r')
dataRussia = Russia_mat['specFeas']
labelRussia = Russia_mat['labels']
dataRussia = np.transpose(dataRussia, (0, 2, 1))
labelRussia = np.transpose(labelRussia, (1, 0)).flatten()
labelRussia = one_hot(labelRussia, outputSize)

acc_ts_TCA_all = np.zeros((numK_Russia, numK_Poland))
kf_R = KFold(n_splits=numK_Russia)
k_Russia = 0
for test_idx, train2_idx in kf_R.split(dataRussia):
    k_Russia += 1
    # if k != kSet:
    #     continue
    train_X_R, train_y_R = dataRussia[train2_idx], labelRussia[train2_idx]
    test_X_R, test_y_R = dataRussia[test_idx], labelRussia[test_idx]

    dataShape = np.shape(train_X_R)
    inputSize = dataShape[1:]
    tca = TCA(kernel_type='linear', dim=8, lamb=1, gamma=1)
    train_X_R_rs = np.reshape(train_X_R, (-1, inputSize[0] * inputSize[1]))
    test_X_R_rs = np.reshape(test_X_R, (-1, inputSize[0] * inputSize[1]))
    data_X_P_rs = np.reshape(dataPoland, (-1, inputSize[0] * inputSize[1]))

    data_X_P_rs_tca, train_X_R_rs_tca, test_X_R_rs_tca = tca.fit_new(data_X_P_rs, train_X_R_rs, test_X_R_rs)

    # data_X_P_tca = np.reshape(data_X_P_rs_tca, (-1, inputSize[0], inputSize[1]))
    # train_X_R_tca = np.reshape(train_X_R_rs_tca, (-1, inputSize[0], inputSize[1]))
    # test_X_R_tca = np.reshape(test_X_R_rs_tca, (-1, inputSize[0], inputSize[1]))
    data_X_P_tca = data_X_P_rs_tca
    train_X_R_tca = train_X_R_rs_tca
    test_X_R_tca = test_X_R_rs_tca

    tensor_trainX_R = torch.Tensor(train_X_R_tca).to(device)  # transform to torch tensor
    tensor_trainY_R = torch.Tensor(train_y_R).to(device)
    train_dataset_R = data.TensorDataset(tensor_trainX_R, tensor_trainY_R)
    train_loader_R = data.DataLoader(train_dataset_R, batch_size=batchSize, shuffle=True)
    tensor_testX_R = torch.Tensor(test_X_R_tca).to(device)  # transform to torch tensor
    tensor_testY_R = torch.Tensor(test_y_R).to(device)
    test_dataset_R = data.TensorDataset(tensor_testX_R, tensor_testY_R)
    test_loader_R = data.DataLoader(test_dataset_R, batch_size=batchSize, shuffle=False)

    kf_P = KFold(n_splits=numK_Poland)
    k_Poland = 0
    for train_idx, val_idx in kf_P.split(data_X_P_tca):
        k_Poland += 1
        # print('training dataset shape: ', train_X.shape)
        # print('training dataset(label) shape: ', train_y.shape)
        # print('testing dataset shape: ', test_X.shape)
        train_X_P, train_y_P = data_X_P_tca[train_idx], labelPoland[train_idx]
        val_X_P, val_y_P = data_X_P_tca[val_idx], labelPoland[val_idx]
        tensor_trainX_P = torch.Tensor(train_X_P).to(device)  # transform to torch tensor
        tensor_trainY_P = torch.Tensor(train_y_P).to(device)
        train_dataset_P = data.TensorDataset(tensor_trainX_P, tensor_trainY_P)
        train_loader_P = data.DataLoader(train_dataset_P, batch_size=batchSize, shuffle=True)
        tensor_valX_P = torch.Tensor(val_X_P).to(device)  # transform to torch tensor
        tensor_valY_P = torch.Tensor(val_y_P).to(device)
        val_dataset_P = data.TensorDataset(tensor_valX_P, tensor_valY_P)
        val_loader_P = data.DataLoader(val_dataset_P, batch_size=batchSize, shuffle=False)

        model_CNN = SpecCNN(inputSize, hiddenSize, outputSize)
        model_CNN.to(device)

        # train_params = [
        #     # model_CNN.fc1.weight, model_CNN.fc1.bias, # ~62
        #     # model_CNN.fc2.weight, model_CNN.fc2.bias, # ~60
        #     # model_CNN.conv1._modules['0'].weight, model_CNN.conv1._modules['0'].bias, # ~68
        #     model_CNN.conv2._modules['0'].weight, model_CNN.conv2._modules['0'].bias,  # ~75
        #     # model_CNN.conv3._modules['0'].weight, model_CNN.conv3._modules['0'].bias # ~72
        # ]
        # optimizer = torch.optim.Adam([{'params': train_params}], lr=LR)
        optimizer = torch.optim.Adam(model_CNN.parameters(), lr=LR)
        scheduler = StepLR(optimizer, step_size=stepLrSize, gamma=0.5)

        acc, acc_val, loss = train(model_CNN, train_loader_P, val_loader_P, numEpoch)

        mat_path = '.\matData\SZ TCA Classification Poland Individual k_P ' + str(k_Poland) + '_' + str(numK_Poland) + \
                   ' k_R ' + str(k_Russia) + '_' + str(numK_Russia) + ' rand_seed ' + str(rand_seed) + '.mat'
        data2saveDic = {"acc": acc,
                        "acc_val": acc_val,
                        "loss": loss}
        io.savemat(mat_path, data2saveDic)

        # To obtain the direct transferring performance (TCA)
        Epoch = np.argmax(acc_val)
        # print('Result in the dataset of Poland: ', acc_val[0, epoch])
        model_pth_pretrained = '.\model_TCA\model Individual -epoch ' + str(Epoch) + '-k_P ' + str(k_Poland) + '_' + str(
            numK_Poland) + '-k_R ' + str(k_Russia) + '_' + str(numK_Russia) + ' -rand_seed ' + str(
            rand_seed) + '.pth'
        model_CNN = SpecCNN(inputSize, hiddenSize, outputSize)
        model_CNN = torch.load(model_pth_pretrained)
        model_CNN.to(device)
        acc_ts_TCA_all[k_Russia - 1, k_Poland - 1] = test(model_CNN, test_loader_R)
    tca_mat_path = '.\matTcaData\SZ TCA Individual  k_R ' + str(k_Russia) + '_' + str(numK_Russia) + ' rand_seed ' + str(rand_seed) + '.mat'
    data2saveDic = {"data_X_P_tca": data_X_P_tca,
                    "data_y_P": labelPoland,
                    "train_X_R_tca": train_X_R_tca,
                    "train_y_R": train_y_R,
                    "test_X_R_tca": test_X_R_tca,
                    "test_y_R": test_y_R}
    io.savemat(tca_mat_path, data2saveDic) # save *_tca variables to avoid recalculate the time-consuming TCA

tca_mat_path2 = '.\matTcaData\SZ TCA acc_ts_TCA_all Individual  k_R ' + str(k_Russia) + '_' + str(numK_Russia) + ' rand_seed ' + str(rand_seed) + '.mat'
data2saveDic = {"acc_ts_TCA_all": acc_ts_TCA_all}
io.savemat(tca_mat_path2, data2saveDic)

print('TCA based results: \n', acc_ts_TCA_all, '\nAverage performacne: ', acc_ts_TCA_all.flatten().mean())

k_Russia = 0
k_Poland = 0
acc_ts_TCA_fineTuning_all = np.zeros((numK_Russia, numK_Poland))
for k_Russia in range(0, numK_Russia):
    k_Russia += 1
    tca_mat_path = '.\matTcaData\SZ TCA Individual  k_R ' + str(k_Russia) + '_' + str(numK_Russia) + ' rand_seed ' + str(rand_seed) + '.mat'
    mat_data_tca = io.loadmat(tca_mat_path)
    train_X_R_tca = mat_data_tca["train_X_R_tca"]
    train_y_R = mat_data_tca["train_y_R"]
    test_X_R_tca = mat_data_tca["test_X_R_tca"]
    test_y_R = mat_data_tca["test_y_R"]
    tensor_trainX_R = torch.Tensor(train_X_R_tca).to(device)  # transform to torch tensor
    tensor_trainY_R = torch.Tensor(train_y_R).to(device)
    train_dataset_R = data.TensorDataset(tensor_trainX_R, tensor_trainY_R)
    train_loader_R = data.DataLoader(train_dataset_R, batch_size=batchSize, shuffle=True)
    tensor_testX_R = torch.Tensor(test_X_R_tca).to(device)  # transform to torch tensor
    tensor_testY_R = torch.Tensor(test_y_R).to(device)
    test_dataset_R = data.TensorDataset(tensor_testX_R, tensor_testY_R)
    test_loader_R = data.DataLoader(test_dataset_R, batch_size=batchSize, shuffle=False)

    dataShape = np.shape(train_X_R_tca)
    inputSize = dataShape[1:]

    for k_Poland in range(0, numK_Poland):

        k_Poland += 1
        mat_path_pretrained = '.\matData\SZ TCA Classification Poland Individual k_P ' + str(k_Poland) + '_' + str(numK_Poland) + \
                   ' k_R ' + str(k_Russia) + '_' + str(numK_Russia) + ' rand_seed ' + str(rand_seed) + '.mat'
        mat_data_pretrained = io.loadmat(mat_path_pretrained)
        acc_val = mat_data_pretrained["acc_val"]
        Epoch = np.argmax(acc_val)
        # print('Result in the dataset of Poland: ', acc_val[0, epoch])
        model_pth_pretrained = '.\model_TCA\model Individual -epoch ' + str(Epoch) + '-k_P ' + str(k_Poland) + '_' + str(
            numK_Poland) + '-k_R ' + str(k_Russia) + '_' + str(numK_Russia) + ' -rand_seed ' + str(rand_seed) + '.pth'
        model_CNN = SpecCNN(inputSize, hiddenSize, outputSize)
        model_CNN = torch.load(model_pth_pretrained)
        model_CNN.to(device)

        train_params = [
            # model_CNN.fc1.weight, model_CNN.fc1.bias,
            # model_CNN.fc2.weight, model_CNN.fc2.bias,
            # model_CNN.conv1._modules['0'].weight, model_CNN.conv1._modules['0'].bias,
            model_CNN.conv2._modules['0'].weight, model_CNN.conv2._modules['0'].bias,
            # model_CNN.conv3._modules['0'].weight, model_CNN.conv3._modules['0'].bias
        ]
        optimizer = torch.optim.Adam([{'params': train_params}], lr=LR)
        # optimizer = torch.optim.Adam(model_CNN.parameters(), lr=LR)
        scheduler = StepLR(optimizer, step_size=stepLrSize, gamma=0.5)

        acc_apt, acc_ts_apt, loss_apt = train(model_CNN, train_loader_R, test_loader_R, numEpoch)
        mat_path = '.\matData\SZ TCA fineTuning Russia Individual -k_P ' + str(k_Poland) + '_' + str(
            numK_Poland) + '-k_R ' + str(k_Russia) + '_' + str(numK_Russia) + ' rand_seed ' + str(rand_seed) + '.mat'
        data2saveDic = {"acc_apt": acc_apt,
                        "acc_ts_apt": acc_ts_apt,
                        "loss_apt": loss_apt}
        io.savemat(mat_path, data2saveDic)
        acc_ts_TCA_fineTuning_all[k_Russia-1, k_Poland-1] = max(acc_ts_apt)

tca_mat_path2 = '.\matTcaData\SZ TCA acc_ts_TCA_fineTuning_all Individual  k_R ' + str(k_Russia) + '_' + str(numK_Russia) + ' rand_seed ' + str(rand_seed) + '.mat'
data2saveDic = {"acc_ts_TCA_fineTuning_all": acc_ts_TCA_fineTuning_all}
io.savemat(tca_mat_path2, data2saveDic)
# print(test(model_CNN, data_loader_Poland))
print('TCA fineTuning results\n', acc_ts_TCA_fineTuning_all, '\nAverage performacne: ', acc_ts_TCA_fineTuning_all.flatten().mean())

