
Experiments:
1. Use HANs with LSTM network to classify sentiment
2. Use HANs with LSTM + attention to classify sentiment

Run the following commands
```
git clone https://github.com/mandalbiswadip/hierarchical-attention-networks.git
cd hierarchical-attention-networks
unzip data.zip
pip install -r requirements.txt
```



### Create dataset

```python
python3 create_training_data.py
```
### Training:

```python
python3 train_attention.py
python3 train_lstm.py
```
### Analyze results
See the Analysis at:
```
analyze attention.ipynb
```

See related blog at: [Self-Attention and HANs](https://mandalbiswadip.github.io/attention/)
