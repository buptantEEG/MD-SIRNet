import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import random
from torchvision import transforms
import time
from torch.utils.tensorboard import SummaryWriter
import sys
from sklearn.metrics import f1_score,accuracy_score,recall_score,precision_score,cohen_kappa_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([transforms.ToTensor()])
class MyDataset(Dataset):
    def __init__(self, txt_path, transform = None, target_transform = None):
        fh = open(txt_path, 'r') 
        imgs = []
        for line in fh:
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], int(words[1])))
            self.imgs = imgs 
            self.transform = transform
            self.target_transform = target_transform
    def __getitem__(self, index):
        fn, label = self.imgs[index] 
        img = Image.open(fn).convert('RGB') 
        if self.transform is not None:
            img = self.transform(img) 
        return img, label
    def __len__(self):
        return len(self.imgs)



class My_Model(nn.Module):
    def __init__(self):
        super(My_Model, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.3)

        self.conv2 = nn.Conv2d(in_channels=128, out_channels=216, kernel_size=7)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(60 * 60 * 216, 48)
        # For binary classification
        # self.output_binary = nn.Linear(48, 2)
        # For multiclass classification
        self.output_multiclass = nn.Linear(48, 3)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.dropout1(x)
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 60 * 60 * 216)
        x = F.relu(self.fc1(x))
        multiclass_output = self.output_multiclass(x)

        return multiclass_output

def seed_torch(seed=1029):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) 
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) 
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

if __name__ == "__main__":

    seed_torch()

    if len(sys.argv) != 3:
        sys.exit(1)
    train_path = sys.argv[1]
    test_path = sys.argv[2]
    
    sub_name = 'xx'

    log_save = './'
    writer = SummaryWriter(log_save)
    print('log_save: ', log_save)

    EPOCH = 100
    BATACH_SIZE = 64
    LEARNING_RATE = 0.0001

    train_loss=0
    train_acc=0
    test_acc=0

    train_data = MyDataset(txt_path=train_path, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATACH_SIZE)
    test_data = MyDataset(txt_path=test_path, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATACH_SIZE)

    criterion = nn.CrossEntropyLoss()
    model = My_Model().cuda(0)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCH):
        model.train()
        tic = time.time()
        acc_train = []
        acc_test = []
        f1_train = []
        f1_test = []
        rec_train = []
        rec_test = []
        prec_train = []
        prec_test = []
        k_train = []
        k_test = []
        for xb, yb in train_loader:    
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            acc_train.append(pred.detach().argmax(1).eq(yb).float().mean().cpu().numpy())
            f1_train.append(f1_score(yb.cpu().numpy(),pred.detach().argmax(1).cpu().numpy(),average='micro'))
            rec_train.append(recall_score(yb.cpu().numpy(),pred.detach().argmax(1).cpu().numpy(),average='micro'))
            prec_train.append(precision_score(yb.cpu().numpy(),pred.detach().argmax(1).cpu().numpy(),average='micro'))
            k_train.append(cohen_kappa_score(yb.cpu().numpy(),pred.detach().argmax(1).cpu().numpy()))
        acc_train = np.mean(acc_train)
        f1_train = np.mean(f1_train)
        rec_train = np.mean(rec_train)
        prec_train = np.mean(prec_train)
        k_train = np.mean(k_train)
        toc = time.time()
        
        with torch.no_grad():
            model.eval()
            for xtest, ytest in test_loader:
                xtest, ytest = xtest.to(device), ytest.to(device)
                pred = model(xtest)
                acc_test.append(pred.detach().argmax(1).eq(ytest).float().mean().cpu().numpy())
                f1_test.append(f1_score(ytest.cpu().numpy(),pred.detach().argmax(1).cpu().numpy(),average='micro'))
                rec_test.append(recall_score(ytest.cpu().numpy(),pred.detach().argmax(1).cpu().numpy(),average='micro'))
                prec_test.append(precision_score(ytest.cpu().numpy(),pred.detach().argmax(1).cpu().numpy(),average='micro'))
                k_test.append(cohen_kappa_score(ytest.cpu().numpy(),pred.detach().argmax(1).cpu().numpy()))
            acc_test = np.mean(acc_test)
            f1_test = np.mean(f1_test)
            rec_test = np.mean(rec_test)
            prec_test = np.mean(prec_test)
            k_test = np.mean(k_test)
        print('Loss at epoch %d : %f, train_acc: %f, test_acc: %f,train_f1: %f,test_f1: %f, train_rec: %f, test_rec: %f,\
              train_prec: %f, test_prec: %f, train_k: %f, test_k: %f, running time: %d'% \
              (epoch, loss.item(), acc_train, acc_test, f1_train,f1_test,rec_train,rec_test,prec_train,prec_test,k_train,k_test,toc-tic))
        writer.add_scalars(main_tag="{}/ACCURACY".format(sub_name), tag_scalar_dict={'train_acc': acc_train, 'test_acc': acc_test}, global_step=epoch)
        writer.add_scalars(main_tag="{}/F1_SCORE".format(sub_name), tag_scalar_dict={'train_f1': f1_train, 'test_f1': f1_test}, global_step=epoch)
        writer.add_scalars(main_tag="{}/REC".format(sub_name), tag_scalar_dict={'train_rec': rec_train, 'test_rec': rec_test}, global_step=epoch)
        writer.add_scalars(main_tag="{}/PREC".format(sub_name), tag_scalar_dict={'train_prec': prec_train, 'test_prec': prec_test}, global_step=epoch)
        writer.add_scalars(main_tag="{}/KAPPA".format(sub_name), tag_scalar_dict={'train_k': k_train, 'test_k': k_test}, global_step=epoch)