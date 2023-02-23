import time
import torch
import pandas as pd
import numpy as np
from utils import dataloader, isCuda, shapeNormalization,  showLayers
from torch.autograd import Variable

temper = 1000
noiseMagnitude1 = 0.0014
criterion = torch.nn.CrossEntropyLoss()

def KNN(df: pd.DataFrame):
    pass
    
def ODIN(nnOutputs: pd.DataFrame):
    # Using temperature scaling
    outputs = outputs / temper


    # Calculating the perturbation we need to add, that is,
    # the sign of gradient of cross entropy loss w.r.t. input
    maxIndexTemp = np.argmax(nnOutputs)
    labels = Variable(torch.LongTensor([maxIndexTemp]).cuda())
    loss = criterion(outputs, labels)
    loss.backward()
    
    # Normalizing the gradient to binary in {0, 1}
    gradient =  (torch.ge(inputs.grad.data, 0))
    gradient = (gradient.float() - 0.5) * 2
    # Normalizing the gradient to the same space of image
    gradient[0][0] = (gradient[0][0] )/(63.0/255.0)
    gradient[0][1] = (gradient[0][1] )/(62.1/255.0)
    gradient[0][2] = (gradient[0][2])/(66.7/255.0)
    # Adding small perturbations to images
    tempInputs = torch.add(inputs.data,  -noiseMagnitude1, gradient)
    outputs = net1(Variable(tempInputs))
    outputs = outputs / temper
    # Calculating the confidence after adding perturbations
    nnOutputs = outputs.data.cpu()
    nnOutputs = nnOutputs.numpy()
    nnOutputs = nnOutputs[0]
    nnOutputs = nnOutputs - np.max(nnOutputs)
    nnOutputs = np.exp(nnOutputs)/np.sum(np.exp(nnOutputs))
    g2.write("{}, {}, {}\n".format(temper, noiseMagnitude1, np.max(nnOutputs)))
    if j % 100 == 99:
        print("{:4}/{:4} images processed, {:.1f} seconds used.".format(j+1-1000, N-1000, time.time()-t0))
        t0 = time.time()

def MSP(df: pd.DataFrame):
    pass

def MDS(df: pd.DataFrame):
    pass

def MLS(df: pd.DataFrame):
    pass

def measure(nn: str, method: str, feature_dataset: str):
    path = f'./features/{feature_dataset}_{nn}_OoD'
    df = pd.read_csv(path)
    
    if method == 'knn':
        raise NotImplementedError(f"{method} not implenented")
        KNN(df)
    if method == 'odin':
        # raise NotImplementedError(f"{method} not implenented")
        ODIN(df)
    if method == 'msp':
        raise NotImplementedError(f"{method} not implenented")
        MSP(df)
    if method == 'mds':
        raise NotImplementedError(f"{method} not implenented")
        MDS(df)
    if method == 'mls':
        raise NotImplementedError(f"{method} not implenented")
        MLS(df)

    # out_datasets = out_datasets.split(' ')
    # path = f'./checkpoints/{checkpoint}'
    # ckpt = torch.load(path)
    # model = torch.hub.load('pytorch/vision:v0.14.0', nn) 
    # use_gpu = isCuda()
    # if use_gpu:
    #     model = model.cuda()

    # model.fc = torch.nn.Identity()
    # missing_keys, unexpected_keys = model.load_state_dict(ckpt['model_state_dict'], strict=False)
    # print(missing_keys)
    # print(unexpected_keys)
    # showLayers(model, (32, 32, 3)) 
    # model.eval()

    # shape, normalization = shapeNormalization(in_dataset, True)
    # testloader = dataloader(in_dataset, normalization, shape[:2], False)
    # output_ID = measure_(method, testloader)

    # output_OoD = {}
    # for out_dataset in out_datasets:
    #     shape, normalization = shapeNormalization(out_dataset)
    #     testloader = dataloader(out_dataset, normalization, shape[:2], False)
    #     output_OoD[out_dataset] = measure_(method, testloader)
    raise NotImplementedError()

