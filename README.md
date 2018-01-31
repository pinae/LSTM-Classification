# LSTM-Classification
Classify texts with a LSTM implemented in Keras

## Installation
Clone this repository:

```bash
git clone git@github.com:pinae/LSTM-Classification.git
cd LSTM-Classification
```

We strongly recommend to use a virtualenv:

```bash
python3 -m venv env
source env/bin/activate
```

Install the dependencies:
```bash
pip install wheel
pip install numpy h5py tensorflow-gpu keras
pip install sacred pymongo
```
Use `tensorflow` instead of `tensorflow-gpu` if you have no 
GPU with CUDA.

## Running experiments
To train a LSTM type:
```bash
python train_lstm.py
```

The experiments use Sacred so you can change parameters on the 
commandline:
```bash
python train_lstm.py with embedding_vector_dimensionality=256 LSTM_dropout_factor=0.3 
```

There are similar experiments for a fully connected network and 
a simple RNN:
```bash
python train_fc.py
python train_simpleRNN.py
```