
import os
import torch
import torchvision
import torch.nn as nn
from PIL import Image
from scipy import stats
import random
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from sklearn.metrics.pairwise import pairwise_distances
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from EMDLoss import EMDLoss
from SoftHistogram import SoftHistogram
from MQuantileLoss_fixedbins import MQuantileLoss
from utils import score_utils
def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)



def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)
    
def EMD(y_true, y_pred):
    cdf_ytrue = np.cumsum(y_true, axis=-1)
    cdf_ypred = np.cumsum(y_pred, axis=-1)
    samplewise_emd = np.sqrt(np.mean(np.square(np.abs(cdf_ytrue - cdf_ypred)), axis=-1))
    return np.mean(samplewise_emd)




class Network(torch.nn.Module):

    def __init__(self, options):
        """Declare all needed layers."""
        nn.Module.__init__(self)
        # Convolution and pooling layers of VGG-16.
        self.features1 = torchvision.models.vgg16(pretrained=True).features#
        self.histogram=SoftHistogram(n_features=512*14*14,n_examples=1,num_bins=3,quantiles=False)
        self.fc = torch.nn.Linear(512*14*14*3, options['numbin'])

    def forward(self, X):
        """Forward pass of the network.
        """
        N = X.size()[0]
        X1 = self.features1(X)
        X1=X1.view(N,-1)
        X = torch.nn.functional.normalize(X1)
        hist = torch.Tensor(N,X.size(1)*3).cuda()
        for i,x in enumerate(X):
            hist[i]=self.histogram(x)
        
        X = self.fc(hist)
        
        X=F.softmax(X,dim=1)
        return X


class NetworkManager(object):
    def __init__(self, options, path):
        """Prepare the network, criterion, solver, and data.
        Args:
            options, dict: Hyperparameters.
        """
        print('Prepare the network and data.')
        self._options = options
        self._path = path

        # Network.
        self._net = torch.nn.DataParallel(Network(self._options), device_ids=[0]).cuda()
        print(self._net)
        self._criterion1 = EMDLoss()
        self._criterion2 = MQuantileLoss()
        # Solver.

        self._solver = torch.optim.Adam(
                self._net.module.parameters(), lr=self._options['base_lr'],
                weight_decay=self._options['weight_decay'])

        
        if (self._options['dataset'] == 'live') :
            crop_size = 448
            train_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize((448,448)),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomCrop(size=crop_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                             std=(0.229, 0.224, 0.225))
            ])
        elif (self._options['dataset'] == 'KONIQ10K') :
            crop_size = 432
            train_transforms = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomCrop(size=crop_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                             std=(0.229, 0.224, 0.225))
            ])

            
            
        test_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize((448,448)),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomCrop(size=crop_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                             std=(0.229, 0.224, 0.225))
        ])
            
           
        if self._options['dataset'] == 'live':  
            # import LIVEFolder
            import LIVEFloder
            train_data = LIVEFloder.LIVEFolder(
                    root=self._path['live'], loader = default_loader, index = self._options['train_index'],
                    transform=train_transforms)
            test_data = LIVEFloder.LIVEFolder(
                    root=self._path['live'], loader = default_loader, index = self._options['test_index'],
                    transform=test_transforms)
        elif self._options['dataset'] == 'KONIQ10K':
            import KONIQ10KFolder
            train_data = KONIQ10KFolder.KONIQ10KFolder(
                    root=self._path['KONIQ10K'], loader = default_loader, index = self._options['train_index'],
                    transform=train_transforms)
            test_data = KONIQ10KFolder.KONIQ10KFolder(
                    root=self._path['KONIQ10K'], loader = default_loader, index = self._options['test_index'],
                    transform=test_transforms)

        self._train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=self._options['batch_size'],
            shuffle=True, num_workers=0, pin_memory=True)
        self._test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=1,
            shuffle=False, num_workers=0, pin_memory=True)

    def train(self):
        """Train the network."""
        print('Training.')
        best_Cosine = 0.0
        best_EMD = 10.0
        best_RMSE = 10.0
        
        best_MOSsrcc = 0.0
        best_MOSplcc = 0.0
        best_MOSrmse = 10.0
        best_epoch = None
        print('Epoch\tTrain loss\tTest_EMD\tTest_RMSE\tTest_Cosine')
        for t in range(self._options['epochs']):
            epoch_loss = []
            for X, y in self._train_loader:
                # Data.
                X = X.cuda()
                y = y.cuda()
                # Clear the existing gradients.
                self._solver.zero_grad()
                # Forward pass.
                score = self._net(X)
                loss1 = self._criterion1(score, y.view(len(score),self._options['numbin']).detach())
                loss2 = self._criterion2(score, y.view(len(score),self._options['numbin']).detach())
                loss=self._options['numbin']*loss1+loss2
                epoch_loss.append(loss.item())
                loss.backward()
                self._solver.step()
            # train_srcc, _ = stats.spearmanr(pscores,tscores)
            EMDtest,RMSEtest,Cosinetest,MOSsrcc,MOSplcc,MOSrmse = self._consitency(self._test_loader)
            if Cosinetest >= best_Cosine:
                best_Cosine = Cosinetest
                
                # if EMDtest <  best_EMD:
                best_EMD = EMDtest
                
                # if RMSEtest < best_RMSE:
                best_RMSE = RMSEtest
                
                # if MOSsrcc >= best_MOSsrcc:
                best_MOSsrcc = MOSsrcc
                    
                # if MOSplcc >= best_MOSplcc:
                best_MOSplcc = MOSplcc
                    
                # if MOSrmse < best_MOSrmse:
                best_MOSrmse = MOSrmse
                best_epoch = t + 1
                print('*', end='')
                pwd = os.getcwd()

                modelpath = os.path.join(pwd,'db_models',('net_params' + '_best' + '.pkl'))
                torch.save(self._net.state_dict(), modelpath)

            print('%d\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f' %
                  (t+1, sum(epoch_loss) / len(epoch_loss),  EMDtest,RMSEtest,Cosinetest,MOSsrcc,MOSplcc,MOSrmse))           

        print('Best at epoch %d, test srcc %f' % (best_epoch, best_Cosine))
        return best_EMD, best_RMSE,best_Cosine, best_MOSsrcc,best_MOSplcc,best_MOSrmse

    def _consitency(self, data_loader):
        self._net.train(False)
        num_total = 0

        EMD_test = []
        EMD_all=0
        EMDtest=0

        RMSE_all=0
        RMSE0=0
        RMSE_test=[]
        RMSEtest=0

        Cosine_all=0
        Cosine0=0
        Cosine_test=[]
        Cosinetest=0
        
                
        pscores_MOS = []
        tscores_MOS = []
        
        for X, y in data_loader:
            # Data.
            X = X.cuda()
            # y = torch.tensor(y.cuda(async=True))
            y = y.cuda()
            # Prediction.
            score = self._net(X)
            score=score[0].cpu()
            y=y[0].cpu()
            pscores_MOS.append(score_utils.mean_score(score.detach().numpy()))
            tscores_MOS.append(score_utils.mean_score(y.detach().numpy()))
            
            ##histogram
            RMSE0=np.sqrt(((score.detach().numpy() - y.detach().numpy()) ** 2).mean())#
            EMD0=EMD(score.detach().numpy(),y.detach().numpy())
            X=[score.detach().numpy(),y.detach().numpy()]
            Cosine0 = (1-pairwise_distances( X, metric='cosine'))[0][1]
            EMD_test.append(EMD0)
            RMSE_test.append(RMSE0)
            Cosine_test.append(Cosine0)

        num_total =len(EMD_test)
        for ele in range(0, len(EMD_test)):
            EMD_all = EMD_all + EMD_test[ele]  
            RMSE_all = RMSE_all + RMSE_test[ele]  
            Cosine_all = Cosine_all + Cosine_test[ele]  
        # EMD_all=torch.sum(EMD_test)
        EMDtest=EMD_all/num_total
        RMSEtest=RMSE_all/num_total
        Cosinetest=Cosine_all/num_total
        
        ##MOS
        MOSsrcc, _ = stats.spearmanr(pscores_MOS,tscores_MOS)
        MOSplcc, _ = stats.pearsonr(pscores_MOS,tscores_MOS)
        MOSrmse=np.sqrt((((pscores_MOS)-np.array(tscores_MOS))**2).mean())
        self._net.train(True)  # Set the model to training phase
        return EMDtest,RMSEtest,Cosinetest,MOSsrcc,MOSplcc,MOSrmse

def main():
    """The main function."""
    import argparse
    parser = argparse.ArgumentParser(
        description='Train DB-CNN for BIQA.')
    parser.add_argument('--base_lr', dest='base_lr', type=float, default=1e-5,
                        help='Base learning rate for training.')
    parser.add_argument('--batch_size', dest='batch_size', type=int,
                        default=8, help='Batch size:8.')
    parser.add_argument('--epochs', dest='epochs', type=int,
                        default=50, help='Epochs for training:50.')
    parser.add_argument('--weight_decay', dest='weight_decay', type=float,
                        default=5e-4, help='Weight decay.')
    parser.add_argument('--dataset',dest='dataset',type=str,default='live',
                        help='dataset: live|KONIQ10K')
    parser.add_argument('--seed',  type=int, default=0)
    
    args = parser.parse_args()
    
    seed = random.randint(10000000, 99999999)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # torch.backends.cudnn.deterministic = True
    print("seed:", seed)
    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed_all(args.seed)
    if args.base_lr <= 0:
        raise AttributeError('--base_lr parameter must >0.')
    if args.batch_size <= 0:
        raise AttributeError('--batch_size parameter must >0.')
    if args.epochs < 0:
        raise AttributeError('--epochs parameter must >=0.')
    if args.weight_decay <= 0:
        raise AttributeError('--weight_decay parameter must >0.')

    options = {
        'base_lr': args.base_lr,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'weight_decay': args.weight_decay,
        'dataset':args.dataset,
        'fc': [],
        'train_index': [],
        'test_index': [],

        'numbin':10
    }
    
    path = {
        # 'live': os.path.join('dataset','databaserelease2'),
        'live': os.path.join('/DATA/gaoyixuan_data/imagehist','LIVE'),
        'KONIQ10K': os.path.join('/DATA/gaoyixuan_data/imagehist','KONIQ10K'),
        'tid2013': os.path.join('dataset','TID2013'),
        'livec': os.path.join('dataset','ChallengeDB_release'),
        'mlive': os.path.join('dataset','LIVEmultidistortiondatabase'),
    }
    
    
    if options['dataset'] == 'live':          
        index = list(range(0,29))
        options['numbin'] == 10
        
    
    
    lr_backup = options['base_lr']
    EMD_all = np.zeros((1,10),dtype=np.float)
    RMSE_all = np.zeros((1,10),dtype=np.float)
    Cosine_all = np.zeros((1,10),dtype=np.float)
    
    MOSsrcc_all = np.zeros((1,10),dtype=np.float)
    MOSplcc_all = np.zeros((1,10),dtype=np.float)
    MOSrmse_all = np.zeros((1,10),dtype=np.float)
    
    for i in range(0,10):
        #randomly split train-test set
        random.shuffle(index)
        train_index = index[0:round(0.8*len(index))]
        test_index = index[round(0.8*len(index)):len(index)]
    
        options['train_index'] = train_index
        options['test_index'] = test_index
    
        #fine-tune all model
        options['base_lr'] = lr_backup
        manager = NetworkManager(options, path)
        EMD, RMSE,Cosine, MOSsrcc,MOSplcc,MOSrmse = manager.train()
        
        EMD_all[0][i] = EMD
        RMSE_all[0][i] = RMSE
        Cosine_all[0][i] = Cosine
        
        #
        MOSsrcc_all[0][i] = MOSsrcc
        MOSplcc_all[0][i] = MOSplcc
        MOSrmse_all[0][i] = MOSrmse
        
    EMD_mean = np.mean(EMD_all)
    RMSE_mean = np.mean(RMSE_all)
    Cosine_mean = np.mean(Cosine_all)
    
    MOSsrcc_mean = np.mean(MOSsrcc_all)
    MOSplcc_mean = np.mean(MOSplcc_all)
    MOSrmse_mean = np.mean(MOSrmse_all)
    # srcc_mean = np.mediam(srcc_all)
    print("seed:", seed)
    print(EMD_all)
    print('average EMD:%4.4f' % (EMD_mean))  
    print(RMSE_all)
    print('average RMSE:%4.4f' % (RMSE_mean))  
    print(Cosine_all)
    print('average Cosine:%4.4f' % (Cosine_mean))  
    
    print(MOSsrcc_all)
    print('average Cosine:%4.4f' % (MOSsrcc_mean))  
    print(MOSplcc_all)
    print('average Cosine:%4.4f' % (MOSplcc_mean))  
    print(MOSrmse_all)
    print('average Cosine:%4.4f' % (MOSrmse_mean))  
    
    return EMD_all,RMSE_all,Cosine_all,MOSsrcc_all,MOSplcc_all,MOSrmse_all


if __name__ == '__main__':
    main()

