import torch
import tensorboardX
from solver import Forest_solver
import os
import torchvision
from dataset import get_loader

if __name__ == "__main__":
    Forest_solver = Forest_solver('../tr_10.txt')

    model_dir = {"checkpoint":"./output/checkpoint/", "tb":"./output/tensorboard/"}
    for dir_ in model_dir:
        if not os.path.exists(model_dir[dir_]):
            os.makedirs(model_dir[dir_])

    train_data = get_loader()
    test_data = get_loader(train=False, batch_size=15)

    writer = tensorboardX.SummaryWriter(model_dir['tb'])

    update_leaf_count = 0

    update_leaf_pred = []
    update_leaf_label = []

    dataiter = iter(train_data)
    testiter = iter(test_data)

    for idx in range(30000):
        #time_start = datetime.datetime.now(asdas
        try:
            data, label = next(dataiter)
        except:
            dataiter = iter(train_data)
            data, label = next(dataiter)

        #torchvision.utils.save_image(data, 'samples.jpg', nrow=4, normalize=True)
        data = data.cuda()
        label = label.cuda()
        loss_item, pred4Pi = Forest_solver.backward_theta(data, label)

        update_leaf_pred.append(pred4Pi)
        update_leaf_label.append(label.view(-1, 1))

        update_leaf_count += 1

        if update_leaf_count >= 50:
            update_leaf_count = 0
            update_leaf_pred = torch.cat(update_leaf_pred, dim=0).transpose(1, 2).detach().cpu().numpy()
            update_leaf_label = torch.cat(update_leaf_label, dim=0).detach().cpu().numpy()

            Forest_solver.backward_pi(update_leaf_pred, update_leaf_label)

            update_leaf_pred = []
            update_leaf_label = []

        

        #learning rate decay
        if (idx+1) % 500 == 0:
            writer.add_scalar('train/loss', loss_item, idx+1)

        if (idx) % 1000 == 0:
            mae = 0.0
            total_num = 0
            kl = 0
            for idx_, (image_t, label_t) in enumerate(test_data):
                total_num += image_t.shape[0]
                image_t = image_t.cuda()
                label_t = label_t.cuda()
                
                pred = Forest_solver.test(image_t)
                #print((label_t - pred).shape)
                #input()
                mae += torch.sum(torch.abs(label_t - pred)).item()
                kl += torch.sum(torch.abs(label_t-pred) < (5/100)).item()
            res_mae = mae /total_num * 100
            res_kl = kl/ total_num
            print("mae: %.4f  cs: %.4f"%(res_mae, res_kl))
            writer.add_scalar('test/mae', res_mae, idx)
            writer.add_scalar('test/cs', res_kl, idx)
        #print(Forest_solver.dist.mean[0])
        #save model
        if (idx)%5000 == 0:
            #
            Forest_solver.save_model(model_dir['checkpoint'], idx + 1)
            #
        #time_end = datetime.datetime.now()i
        Forest_solver.update_learning_rate()
        print('[%d/%d] loss: %.4f'% (idx+1, 30000, loss_item))
        #print('alpha: ', model.alpha)
        #print("remains {:.4f} minutes...".format((time_end - time_start).total_seconds() / 60. * (max_step - idx)))
