import argparse
import cv2
import numpy as np
import torch
from torch.autograd import Function
from torchvision import models
import torch.nn.functional as F
from solver import Forest_solver
from dataset import get_loader
from torch.autograd import Variable

class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.max_pool_layers = [1, 3, 6, 9, 12]
        self.fc_layer_idx = [12]
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def get_gradients(self):
        return self.gradients

    def __call__(self, x):
        target_activations = []
        for idx, (name, module) in enumerate(self.model._modules.items()):
            if module == self.target_layers:
                print('ok!!!!!!!!!!!')
                x = F.relu(module(x))
                target_activations = x
                x.register_hook(self.save_gradient)
            else:
                x = F.relu(module(x))

            if idx in self.max_pool_layers:
                x = F.max_pool2d(x, 2, 2)

            if idx in self.fc_layer_idx:
                x = x.view(x.size(0), -1)    

        
        return target_activations, x[:, :128]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='DRF')
    parser.add_argument('--tree_id', type=int, required=False, default=0)
    parser.add_argument('--node_id', type=int, required=False, default=1)
    args=parser.parse_args()

    
    Forest_solver = Forest_solver('../tr_10.txt')
    Forest_solver.load_model('./output/checkpoint/', 25001)
    Forest_solver.model.eval()

    F_extractor = ModelOutputs(Forest_solver.model.feature_layer, Forest_solver.model.feature_layer.conv_5_3)

    test_data = get_loader(train=False, batch_size=1)
    test_iter = iter(test_data)
    data, label = next(test_iter)
    data = data.cuda()
    target_activations, vgg_out = F_extractor(data)

    tree = Forest_solver.model.forest.trees[args.tree_id]

    node_id = args.node_id

    

    feats = torch.mm(vgg_out, tree.feature_mask.cuda())
    feats = torch.sigmoid(feats) #
    decision = torch.unsqueeze(feats,dim=2)
    decision_comp = 1-decision
    decision = torch.cat((decision,decision_comp),dim=2) # -> [batch_size,n_leaf,2]

    # compute route probability
    batch_size = decision.size()[0]
    _mu = Variable(vgg_out.data.new(batch_size,1,1).fill_(1.))
    begin_idx = 0
    end_idx = 1

    for n_layer in range(0, 6 - 1):
        _mu = _mu.view(batch_size,-1,1).repeat(1,1,2)
        _decision = decision[:, begin_idx:end_idx, :]  # -> [batch_size,2**n_layer,2]
        _mu = _mu*_decision # -> [batch_size,2**n_layer,2]
        if node_id >= begin_idx and node_id < end_idx:
            break
        begin_idx = end_idx
        end_idx = begin_idx + 2 ** (n_layer+1)

    import math
    cur_layer = math.floor(np.log2(node_id+1))
    cur_id = node_id - (2 ** cur_layer) + 1

    final = _mu[:, :, 0]
    one_hot = np.zeros((1, final.size()[-1]), dtype=np.float32)
    one_hot[0][cur_id] = 1
    one_hot = torch.from_numpy(one_hot).requires_grad_(True)
    one_hot = torch.sum(one_hot.cuda() * final)

    Forest_solver.model.feature_layer.zero_grad()
    Forest_solver.model.forest.zero_grad()
    one_hot.backward(retain_graph=True)

    grads_val = F_extractor.get_gradients()[-1].cpu().data.numpy()

    target = target_activations
    target = target.cpu().data.numpy()[0, :]

    weights = np.mean(grads_val, axis=(2, 3))[0, :]
    cam = np.zeros(target.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * target[i, :, :]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, data.shape[2:])
    cam = cam - np.min(cam)
    cam = cam / np.max(cam)

    img = data.permute(0, 2, 3, 1).cpu().numpy()[0][...,::-1]
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    #print(np.uint8(255 * img).max(), np.uint8(255 * img).min(), cam.shape)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cv2.imwrite("cam.jpg", np.uint8(255 * cam))
    


    







