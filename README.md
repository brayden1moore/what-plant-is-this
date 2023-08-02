# What Plant Is This?

Welcome to the WPiT repository. This project is an image recognition model specifically designed for identifying plants. It utilizes a Google Vision Transformer feature extractor with downstream dense layers to achieve accurate plant classification suggestions.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Training](#training)
- [Contributing](#contributing)
- [License](#license)

## Introduction

I've always been terrible at identifying plants. And when you've got to walk through 50 yards of tall brush to get to your camping spot, it's good to know if it's Queen Anne's Lace or Poison Parsnip that's about to brush up against your leg.<br><br>
Thanks to the [Global Biodiversity Information Facility](https://www.gbif.org) and their Occurrence database, I was able to train a vision model on ~80k plant images (augmented with Torchvision transforms) from the U.S. and Canada, labeled with their species and feature (leaf, flower, or fruit).<br><br>
There are around 3,800 unique species in the data -- some of which may not have made it into the training set -- which makes classification challenging. To increase usability, I opted to have the app return the top 9 predictions. In testing, the correct species is within these 9 predictions for 79% of flowers, 78% of fruits, and 60% of leaves. I think this makes sense, since there is much more variability in color among flowers and fruits than leaves.<br><br>
Try it out [here](https://www.braydenmoore.com/plant).

## Features

- Google Vision Transformer feature extractor.
- Convolutional Neural Network (CNN).
- Pre-trained model for direct usage or as a starting point for further training.
- Python scripts to train and test.
- Web app for interactive plant image recognition.

## Installation

1. Clone this repository to your local machine using:

```bash
git clone https://github.com/brayden1moore/What-Plant-Is-This.git
cd What-Plant-Is-This
```

2. Set up a virtual environment (optional but recommended) and activate it:

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

To launch the web app locally:

```bash
python app.py
```

I will be adding a script shortly to make predictions via the command line.

## Training

If you want to retrain any of the three models (leaf, flower, fruit), you can use the provided Python script `train.py`. It will train until test accuracy reaches a maximum.

You can then initiate the training process with the following command:

```bash
python train.py -f flower -s 0.75 -b 16 -scratch False
```
`-f` specifies which plant feature you are training on. If blank, will train all<br>
`-s` train split (default 0.75).<br>
`-b` batch size (default 16).<br>
`-scratch` if set True will retrain from the beginning (default False).<br><br>

Please feel free to tweak the hyperparameters and network architecture as per your requirements.

## Contributing

Contributions to this project are always welcome. If you find any issues or want to suggest improvements, please open an issue or submit a pull request. For major changes, kindly open an issue to discuss your ideas beforehand.

## License

This project is licensed under the [MIT License](LICENSE).

---

Thanks for reading!
