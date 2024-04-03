from torch_lib.Dataset import *
from torch_lib.Model import Model, OrientationLoss
from repvgg_pytorch import get_RepVGG_func_by_name
from repvgg_pytorch import repvgg_model_convert
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils import data
from ultralytics import YOLO
import torch.onnx
import os

def main():

    # hyper parameters
    epochs = 100
    batch_size = 2
    #alpha = 0.4
    w = 0.1

    print("Loading all detected objects in dataset...")

    train_path = os.path.abspath(os.path.dirname(__file__)) + '/Kitti/training'
    dataset = Dataset(train_path)
    params = {'batch_size': batch_size,
              'shuffle': True,
              'num_workers': 6}

    generator = data.DataLoader(dataset, **params)


    model = Model(model_name = 'RepVGG-A0', deploy =False).cuda()
    opt_SGD = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
    conf_loss_func = nn.CrossEntropyLoss().cuda()
    dim_loss_func = nn.MSELoss().cuda()
    orient_loss_func = OrientationLoss
    Alpha_loss_func = nn.SmoothL1Loss().cuda()
    # load any previous weights
    model_path = os.path.abspath(os.path.dirname(__file__)) + '/weights/'
    latest_model = None
    first_epoch = 0
    if not os.path.isdir(model_path):
        os.mkdir(model_path)
    else:
        try:
            latest_model = [x for x in sorted(os.listdir(model_path)) if x.endswith('.pkl')][-1]
        except:
            pass


    if latest_model is not None:
        checkpoint = torch.load(model_path + latest_model)
        model.load_state_dict(checkpoint['model_state_dict'])
        opt_SGD.load_state_dict(checkpoint['optimizer_state_dict'])
        first_epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print('Found previous checkpoint: %s at epoch %s'%(latest_model, first_epoch))
        print('Resuming training....')



    total_num_batches = int(len(dataset) / batch_size)

    for epoch in range(first_epoch+1, epochs+1):
        curr_batch = 0
        passes = 0
        for local_batch, local_labels in generator:

            #truth_orient = local_labels['Orientation'].float().cuda()
            truth_conf = local_labels['Confidence'].long().cuda()
            #Alpha = local_labels['Alpha'].float().cuda()
            #truth_dim = local_labels['Dimensions'].float().cuda()
            local_batch=local_batch.float().cuda()
            conf = model(local_batch)
            #orient_loss = orient_loss_func(orient, truth_orient, truth_conf)
            #dim_loss = dim_loss_func(dim, truth_dim)
            #print(truth_conf)
            truth_conf = torch.max(truth_conf, dim=1)[1]
            conf_loss = conf_loss_func(conf, truth_conf)
            loss_theta = conf_loss
            #loss = Alpha_loss_func(alpha, Alpha)
            loss = loss_theta
            opt_SGD.zero_grad()
            loss.backward()
            opt_SGD.step()


            if passes % 100 == 0:
                print("--- epoch %s | batch %s/%s --- [loss: %s]" %(epoch, curr_batch, total_num_batches, loss.item()))
                passes = 0
                #print(f"truth_orient:{truth_orient} | predicted: {orient}")
                #print(f"truth_conf:{truth_conf} | predicted: {conf}")


            passes += 1
            curr_batch += 1

        # save after every 10 epochs
        if epoch % 10 == 0:
            name = model_path + 'epoch_%s.pkl' % epoch
            onnx_name = model_path + 'epoch_%s.onnx' % epoch
            print("====================")
            print ("Done with epoch %s!" % epoch)
            print ("Saving weights as %s ..." % name)
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': opt_SGD.state_dict(),
                    'loss': loss
                    }, name)
            print("====================")
            print("=====ONNX format=====")
            input_tensor = torch.zeros([1, 3, 224, 224]).cuda()
            torch.onnx.export(model, input_tensor, onnx_name, input_names=["input"], output_names=["confidence"])
            #torch.onnx.export(model, input_tensor, onnx_name, input_names=["input"], output_names=["alpha"])
            print("=====ONNX done=====")
        

if __name__=='__main__':
    main()