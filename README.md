## Word2vec custom implementation
### Author: Szymon Pobłocki
The variant implemented is n-gram with negative sampling.

### How to use
Prepare env (tested on Python 3.12.3):
```
python3 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt
```

To train the model:
python main.py

To test the model:
python test.py

### Gradient derivations
The loss formula is:
$$L = - \log(\sigma(v_0))