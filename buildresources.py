import pandas as pd
import pickle as pkl 
import torch 
import numpy as np
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as T

print('Loading data...')
with open('app/resources/infoDict.pkl','rb') as f:
    infoDict = pkl.load(f)

for feature in ["leaf","flower","fruit"]:
    print(f'{feature}...')
    infoDf = pd.DataFrame(infoDict).T
    infoDf = infoDf.loc[infoDf['feature']==feature]
    counts = infoDf.groupby('species').agg({'identifier':'nunique'}).reset_index()
    speciesSelections = counts.loc[counts['identifier']>=5, 'species'].values
    infoDf = infoDf.loc[infoDf['species'].isin(speciesSelections)]

    gbifIDList = infoDf['gbif'].values
    speciesList = infoDf['species'].values
    imageIndices = list(infoDf.index)

    imageSpeciesDict = dict(zip(imageIndices,speciesList))
    labelSet = list(set(speciesList))

    imageLabels = {}
    for i in imageIndices:
        name = imageSpeciesDict[i]
        imageLabels[i] = labelSet.index(name)

    with open(f'app/resources/{feature}LabelSet.pkl', 'wb') as f:
        pkl.dump(labelSet, f)

    with open(f'app/resources/{feature}ImageIndices.pkl', 'wb') as f:
        pkl.dump(imageIndices, f)

    with open(f'app/resources/{feature}ImageLabels.pkl', 'wb') as f:
        pkl.dump(imageLabels, f)

    print('Done getting indices and labels')
    print(len(imageIndices), 'samples')
    print(len(labelSet), 'classes')

print('Getting mean and standard deviation RGB')

for feature in  ['leaf','flower','fruit']:
    with open(f'app/resources/{feature}ImageIndices.pkl','rb') as f:
        imageIndices = pkl.load(f)

    rMeans = []
    gMeans = []
    bMeans = []

    rStds = []
    gStds = []
    bStds = []

    for i in tqdm(imageIndices):
        convert = T.ToTensor()
        img = convert(Image.open(fr'app/images/img{i}.jpeg'))
        r, g, b = torch.mean(img, dim=[1,2])
        rMeans.append(r.item())
        gMeans.append(g.item())
        bMeans.append(b.item())

        r, g, b = torch.std(img, dim=[1,2])
        rStds.append(r.item())
        gStds.append(g.item())
        bStds.append(b.item())

    rM = np.array(rMeans).mean()
    gM = np.array(gMeans).mean()
    bM = np.array(bMeans).mean()
    rS = np.array(rStds).mean()
    gS = np.array(gStds).mean()
    bS = np.array(rStds).mean()

    meansAndStds = {'mean' : [rM,gM,bM],
            'std': [rS,gS,bS]}

    with open(f'app/resources/{feature}MeansAndStds.pkl', 'wb') as f:
        pkl.dump(meansAndStds, f)
    