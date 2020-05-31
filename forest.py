
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from collections import OrderedDict
import numpy as np
import torch.nn.functional as F
import pickle

from utils.gaussian import gaussian_func

class Tree(nn.Module):
    def __init__(self,depth,n_in_feature):
        super(Tree, self).__init__()
        self.depth = depth
        self.n_leaf = 2 ** (depth - 1)

        # used features in this tree
        n_used_feature = self.n_leaf - 1
        onehot = np.eye(n_in_feature)
        using_idx = np.random.choice(np.arange(n_in_feature), n_used_feature, replace=False)
        self.feature_mask = onehot[using_idx].T
        self.feature_mask = Parameter(torch.from_numpy(self.feature_mask).type(torch.FloatTensor),requires_grad=False)


    def forward(self,x):
        """
        :param x(Variable): [batch_size,n_features]
        :return: route probability (Variable): [batch_size,n_leaf]
        """
        if x.is_cuda and not self.feature_mask.is_cuda:
            self.feature_mask = self.feature_mask.cuda()

        feats = torch.mm(x,self.feature_mask) # ->[batch_size,n_used_feature]
        decision = torch.sigmoid(feats) # ->[batch_size,n_leaf]

        decision = torch.unsqueeze(decision,dim=2)
        decision_comp = 1-decision
        decision = torch.cat((decision,decision_comp),dim=2) # -> [batch_size,n_leaf,2]

        # compute route probability
        batch_size = x.size()[0]
        _mu = Variable(x.data.new(batch_size,1,1).fill_(1.))
        begin_idx = 0
        end_idx = 1
        for n_layer in range(0, self.depth - 1):
            _mu = _mu.view(batch_size,-1,1).repeat(1,1,2)
            _decision = decision[:, begin_idx:end_idx, :]  # -> [batch_size,2**n_layer,2]
            _mu = _mu*_decision # -> [batch_size,2**n_layer,2]
            begin_idx = end_idx
            end_idx = begin_idx + 2 ** (n_layer+1)

        mu = _mu.view(batch_size,self.n_leaf)
        #print(mu[:, :5])
        return mu



class Forest(nn.Module):
    def __init__(self,n_tree,tree_depth,n_in_feature):
        super(Forest, self).__init__()
        self.trees = nn.ModuleList()
        self.n_tree  = n_tree
        self.dist = Pi(n_tree, tree_depth)
        for _ in range(n_tree):
            tree = Tree(tree_depth,n_in_feature)
            self.trees.append(tree)

    def forward(self,x):
        probs = []
        for tree in self.trees:
            mu = tree(x)
            probs.append(mu.unsqueeze(2))
        pi = self.dist.get_mean()
        probs = torch.cat(probs,dim=2) # bs, 32, 5
        prob = probs * pi.transpose(0, 1).unsqueeze(0)     
        prob = torch.sum(prob, dim=1) 
        return prob, probs




class NeuralDecisionForest(nn.Module):
    def __init__(self, feature_layer, forest):
        super(NeuralDecisionForest, self).__init__()
        self.feature_layer = feature_layer
        self.forest = forest

    def forward(self, x):
        out = self.feature_layer(x)
        out = out.view(x.size()[0],-1)
        out = self.forest(out)
        return out

class Pi():
    def __init__(self, num_tree, tree_depth, task_num=1,\
                    iter_num=20, samples=50*16):

        leaf_node_per_tree = 2 ** (tree_depth - 1)
        self.mean = np.random.rand(num_tree, leaf_node_per_tree, task_num, 1).astype(np.float32)
        self.sigma = np.random.rand(num_tree, leaf_node_per_tree, task_num, task_num).astype(np.float32)
        self.iter_num = iter_num
        self.samples = samples
    def init_kmeans(self, mean, sigma): #mean, sigma -> ndarray
        # implement only for task_num=1
        n_t, leaf_n, t_num, _ = self.mean.shape
        for i in range(leaf_n):
            self.mean[:, i, :, :] = mean[i]
            self.sigma[:, i, :, :] = sigma[i]

    def get_mean(self, cuda=True):
        if cuda:
            return torch.tensor(self.mean).squeeze().cuda()
        else:
            return torch.tensor(self.mean).squeeze()

    def update(self, x, y):
        """
         x has the shape of [samples, num_tree, leaf_num],
         y hsa the shape of [samples, 1]
         gaussian_function will return a probability \\
          array with shape of [samples, num_tree, leaf_num].

        """
        num_tree, leaf_num, _, _  = self.mean.shape
        #print(y.shape)
        for i in range(self.iter_num):

            gaussian_value = gaussian_func(y, self.mean, self.sigma) # [samples, num_tree, leaf_num]

            all_leaf_prob_pi = x * (gaussian_value + 1e-9) # [samples, num_tree, leaf_num]
            all_leaf_sum_prob = np.sum(all_leaf_prob_pi, axis=2, keepdims=True)  #[samples, num_tree, 1]

            zeta = all_leaf_prob_pi / (all_leaf_sum_prob + 1e-9) # [samples, num_tree, leaf_num]

            y_temp = np.expand_dims(y, 2)
            y_temp = np.repeat(y_temp, num_tree, 1)
            y_temp = np.repeat(y_temp, leaf_num, 2)

            zeta_y = zeta * y_temp # [samples, num_tree, leaf_num]
            zeta_y = np.sum(zeta_y, 0) # [num_tree, leaf_num]
            zeta_sum  = np.sum(zeta, 0) # [num_tree, leaf_num]

            mean = zeta_y / (zeta_sum + 1e-9)

            self.mean[:,:, 0, 0] = mean

            mean_new = y_temp - np.expand_dims(mean, 0) # [samples, num_tree, leaf_num]

            zeta_for_sigma = zeta * mean_new * mean_new
            zeta_for_sigma = np.sum(zeta_for_sigma, 0)

            sigma = zeta_for_sigma / (zeta_sum + 1e-9)

            self.sigma[:,:,0,0] = sigma
            #print(mean)
            #input()

    def save_model(self, path, epoch):
        with open(path + 'pi_' + str(epoch),'wb') as f:
            pickle.dump(self.mean, f)
            pickle.dump(self.sigma, f)

    def load_model(self, path, epoch):
        with open(path + 'pi_' + str(epoch) ,'rb') as f:
            self.mean = pickle.load(f)
            self.sigma = pickle.load(f)
