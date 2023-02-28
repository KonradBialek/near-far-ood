import torch
import pandas as pd
import numpy as np
from .utils import dataloader, getNN, isCuda, getShape, getNormalization, loadNNWeights, save_scores,  showLayers
import faiss

temper = 1000
noiseMagnitude1 = 0.0014
criterion = torch.nn.CrossEntropyLoss()
normalizer = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True) + 1e-10

def KNN(data: pd.DataFrame, method_args):
    K = int(method_args[0])
    data = data.to_numpy()
    activation_log = normalizer(data.reshape(
                    data.shape[0], data.shape[1], -1).mean(2))

    index = faiss.IndexFlatL2(data.shape[1])
    activation_log = np.ascontiguousarray(activation_log)
    index.add(np.float32(activation_log))

    feature_normed = np.float32(np.ascontiguousarray(normalizer(data)))
    D, _ = index.search(
        feature_normed,
        K,
    )
    return torch.from_numpy(D[:, -1])
    
# def ODIN(nnOutputs: pd.DataFrame):
#     # Using temperature scaling
#     outputs = outputs / temper


    # # Calculating the perturbation we need to add, that is,
    # # the sign of gradient of cross entropy loss w.r.t. input
    # maxIndexTemp = np.argmax(nnOutputs)
    # labels = Variable(torch.LongTensor([maxIndexTemp]).cuda())
    # loss = criterion(outputs, labels)
    # loss.backward()
    
    # # Normalizing the gradient to binary in {0, 1}
    # gradient =  (torch.ge(inputs.grad.data, 0))
    # gradient = (gradient.float() - 0.5) * 2
    # # Normalizing the gradient to the same space of image
    # gradient[0][0] = (gradient[0][0] )/(63.0/255.0)
    # gradient[0][1] = (gradient[0][1] )/(62.1/255.0)
    # gradient[0][2] = (gradient[0][2])/(66.7/255.0)
    # # Adding small perturbations to images
    # tempInputs = torch.add(inputs.data,  -noiseMagnitude1, gradient)
    # outputs = net1(Variable(tempInputs))
    # outputs = outputs / temper
    # # Calculating the confidence after adding perturbations
    # nnOutputs = outputs.data.cpu()
    # nnOutputs = nnOutputs.numpy()
    # nnOutputs = nnOutputs[0]
    # nnOutputs = nnOutputs - np.max(nnOutputs)
    # nnOutputs = np.exp(nnOutputs)/np.sum(np.exp(nnOutputs))
    # g2.write("{}, {}, {}\n".format(temper, noiseMagnitude1, np.max(nnOutputs)))
    # if j % 100 == 99:
    #     print("{:4}/{:4} images processed, {:.1f} seconds used.".format(j+1-1000, N-1000, time.time()-t0))
    #     t0 = time.time()

def MSP(df: pd.DataFrame):
    pass

def MDS(df: pd.DataFrame):
    pass

def MLS(df: pd.DataFrame):
    pass

def measure(nn: str, method: str, feature_datasets: list, method_args: list):
    for i, dataset in enumerate(feature_datasets):
        if i == 0:
            file = f'{dataset}_{nn}_ID.csv'
        else:
            file = f'{dataset}_{nn}_OoD.csv'
        path = f'./features/{file}'
        data = pd.read_csv(path)
        label = data.pop('class')
        # model = loadNNWeights(nn, checkpoint)
        # postprocessor = get_postprocessor(method, method_args)

        if method == 'knn':
            output = KNN(data, method_args)
            # print(output)

        # if method == 'odin':
        #     # raise NotImplementedError(f"{method} not implenented")
        #     ODIN(df)
        # if method == 'msp':
        #     raise NotImplementedError(f"{method} not implenented")
        #     MSP(df)
        # if method == 'mds':
        #     raise NotImplementedError(f"{method} not implenented")
        #     MDS(df)
        # if method == 'mls':
        #     raise NotImplementedError(f"{method} not implenented")
        #     MLS(df)

        data["class"] = label
        data[method] = output
        # print(data)
        data.to_csv(path[:-4]+'_distance.csv', mode='w', index=False, header=True)

    raise NotImplementedError()
