
Experiments:
1. Use HANs with LSTM network to classify sentiment
2. Use HANs with LSTM + attention to classify sentiment

Run the following commands
```
git clone https://github.com/mandalbiswadip/hierarchical-attention-networks.git
cd hierarchical-attention-networks
unzip data.zip
```

### Training:

```
python3 train_attention.py
python3 train_lstm.py
```
#### train lstm
Takes around 120s for training a single epoch on  Tesla P100 GPU