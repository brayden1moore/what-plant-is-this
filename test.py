import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-f', type=str, help='Plant feature: "leaf","flower","fruit"')
parser.add_argument('-b', type=int, help='Batch size')
parser.add_argument('-k', type=int, help='Top K predictions (for testing)')
args = parser.parse_args()

features = ([args.f] if args.f is not None else ['fruit','leaf','flower'])
batchSize = (args.b if args.b is not None else 16)
k = (args.k if args.k is not None else 9)

from plantnet import PlantNet
import random
import torch
import torchvision.transforms as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle as pkl
import numpy as np
from torchvision.io import read_image
from tqdm import tqdm

for feature in features:
    print(f'{feature.title()}')
    with open(f'resources/{feature}LabelSet.pkl', 'rb') as f:
        labelSet = pkl.load(f)

    with open(f'resources/{feature}ImageLabels.pkl', 'rb') as f:
        imageLabels = pkl.load(f)

    with open(f'resources/{feature}ImageIndices.pkl','rb') as f:
        imageIndices = pkl.load(f)

    with open(f'resources/{feature}MeansAndStds.pkl', 'rb') as f:
        meansAndStds = pkl.load(f)

    preprocess = torch.nn.Sequential( 
    T.CenterCrop(224),
    T.RandomHorizontalFlip(), 
    T.RandomVerticalFlip(), 
    T.RandomAutocontrast(),
    T.ConvertImageDtype(torch.float32),
    T.Normalize(
        mean=meansAndStds['mean'],
        std=meansAndStds['std']
    ))

    def stackImages(index, batchSize, imageIndices):
        start = index * batchSize
        end = (index + 1) * batchSize
        iter = start
        tensors = []
        while iter<end:
            image = read_image(f'images-highres/img{imageIndices[iter]}.jpeg').to(device)
            tensors.append(preprocess(image))
            iter += 1
        return torch.stack(tensors)

    def stackLabels(index, batchSize, imageIndices):
        start = index * batchSize
        end = (index + 1) * batchSize
        iter = start
        tensors = []
        while iter<end:
            tensors.append(torch.tensor(imageLabels[imageIndices[iter]]).to(device))
            iter += 1
        return torch.stack(tensors)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(labelSet) 

    model = PlantNet(num_classes=len(labelSet))
    model.load_state_dict(torch.load(fr"models/plantnet-model-{feature}.pt"))
    model.to(device)

    with open(f'batches/{feature}TrainBatches.pkl','rb') as f:
        trainBatches = pkl.load(f)
    with open(f'batches/{feature}TestBatches.pkl', 'rb') as f:
        testBatches = pkl.load(f)
    
    testAccuracies = []
    epoch = 1
    keepGoing = True


    model.eval()
    correct = []
    with torch.no_grad():
        for j in tqdm(testBatches):
            images = stackImages(j,batchSize,imageIndices).to(device)
            labels = stackLabels(j,batchSize,imageIndices).to(device)

            outputs = model(images).to(device)
            top = torch.topk(outputs,k,dim=1)
            predictions = top.indices.to(device)

            for p,l in zip(predictions,labels):
                correct.append(int(l in p))

    print(f'Test accuracy: {np.array(correct).mean()*100}%')