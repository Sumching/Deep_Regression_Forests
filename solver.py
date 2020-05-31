
from networks.vgg import vgg16_bn
from utils.kmeans import kmeans
from torch.optim import lr_scheduler
import torch
import torch.nn as nn
import numpy as np
import pickle

import forest as ndf





class Forest_solver():
    def __init__(self, list_dir, num_tree=5, depth=6, task_num=1):
        #super(Forest, self).__init__()
        feat_layer = vgg16_bn(num_classes=128)

        forest = ndf.Forest(n_tree=num_tree, tree_depth=depth, n_in_feature=128)
        model = ndf.NeuralDecisionForest(feat_layer, forest)

        self.leaf_node_num = 2 ** (depth - 1)
        model = model.cuda()
        self.model = model
        #self.dist = Pi(num_tree, self.leaf_node_num)
        init_mean, init_sigma = self.kmeans_label(list_dir)
        self.model.forest.dist.init_kmeans(init_mean, init_sigma)
        #self.optimizer = torch.optim.Adam(self.model.parameters(),lr=0.05, betas=(0.5, 0.999))

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.05, momentum=0.9)
        self.optimizers = [self.optimizer]
        def lambda_rule(iteration):
            if iteration < 10000:
                lr_l = 1
            if 10000 <= iteration and iteration < 20000:
                lr_l = 0.5
            if iteration >= 20000:
                lr_l = 0.25
            return lr_l
        self.schedulers = [lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule) for optimizer in self.optimizers]


    def kmeans_label(self, list_dir):
        labels = []
        with open(list_dir,"r") as f: 
            lines = f.readlines()      
            for line in lines:
                if 'noise' not in line:
                    label = line.strip('\n').split(' ')[-1]
                    #print(label)
                    label = int(label) / 100.0
                    labels.append(label)

        labels = np.reshape(np.array(labels), [-1, 1])
        #print(labels.shape, labels.max(), labels.min(), self.leaf_node_num)
        init_mean, init_sigma = kmeans(labels, self.leaf_node_num)
        #print(init_mean, init_sigma)
        #input()
        return init_mean, init_sigma

    def forward(self, x):
        predictions, pred4Pi = self.model(x)
        #input()
        print(torch.mean(predictions, dim=1, keepdim=True))
        return predictions, pred4Pi
    def get_loss(self, x, y):
        predictions, pred4Pi = self.forward(x)
        print(y)
        loss = torch.sum(0.5 * (y.view(-1, 1) - predictions) ** 2)/x.shape[0]
        return loss, pred4Pi

    def test(self, x):
        self.model.eval()
        pred, _ = self.forward(x)
        return torch.mean(pred, dim=1)

    def backward_theta(self, x, y):
        self.model.train()
        self.optimizer.zero_grad()
        loss, pred4Pi = self.get_loss(x, y)
        
        loss.backward()
        self.optimizer.step()

        return loss.item(), pred4Pi

    def backward_pi(self, x, y):
        self.model.forest.dist.update(x, y)
        #print(self.model.forest.dist.get_mean())
        #input()

    def save_model(self, path, epoch):
        torch.save(self.model.state_dict(), path + 'model_{}.pth'.format(epoch))

        self.model.forest.dist.save_model(path, epoch)

    def load_model(self, path, epoch):
        self.model.load_state_dict(torch.load(path + 'model_{}.pth'.format(epoch)))

        self.model.forest.dist.load_model(path, epoch)

    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)


        

                

