# Word2vec: numpy implementation
## Author: Szymon Pobłocki
The goal of this project was to prepare a pure numpy custom version of classic word2vec technique. The variant implemented is n-gram with negative sampling. Dataset used is **text8**, popular in that context, which is a cleaned 100MB slice of English Wikipedia. Parameters used for testing are set as defaults in the main function in the ```main.py``` file.

## How to use
Prepare env (tested on Python 3.12.3):
```
python3 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt
```

To train the model:
```python main.py```

To test the model:
```python test.py```

## Example outputs
Below are the example results of my trained model. It was trained for one epoch (one pass with sliding window over whole corpus).

Top 5 most similar results for word *king* (together with similarity):
  - *emperor*: 0.9522 
  - *prince*: 0.9491 
  - *portugal*: 0.9458 
  - *frederick*: 0.9452
  - *sultan*: 0.9433

Top 5 most similar results for word *capital* (together with similarity):
  - *affairs*: 0.9786
  - *communist*: 0.9753
  - *azerbaijan*: 0.9748
  - *security*: 0.9743
  - *transportation*: 0.9743
## Gradient derivations
The loss formula is:

$$ L = - \log(\sigma(v_o^\top v_c)) - \sum_i \log(\sigma(-v_{n_i}^\top v_c)) $$

where $v_c$ is the embedding of the center of our window, $v_o$ is the embedding of word from the window, $v_{n_i}$'s are the negative embeddings (from random words outside of the window).

We don't have to go all the way back to the projections - we are only using one particular row of the matrix for each of them, which simplifies the formulas.

#### Loss with respect to $v_o$
Using the chain rule:

$$\frac{\partial{L}}{\partial{v_o}} = \frac{\partial{L}}{\partial{ \log(\sigma(v_o^\top v_c))}} \cdot \frac{\partial{ \log(\sigma(v_o^\top v_c))}}{\partial{\sigma(v_o^\top v_c))}} \cdot \frac{\partial{\sigma(v_o^\top v_c))}}{\partial{v_o}}$$

$$\frac{\partial{L}}{\partial{v_o}} = -1 \cdot \frac{1}{\cancel{\sigma(v_o^\top v_c))}} \cdot \sigma(\cancel{v_o^\top v_c})(1-\sigma(v_o^\top v_c))\cdot v_c = (\sigma(v_o^\top v_c) - 1)\cdot v_c$$

#### Loss with respect to $v_{n_i}$
Similar computation to the one above, only the sign differs:

$$\frac{\partial{L}}{\partial{v_{n_i}}}= - (\sigma(-v_{n_i}^\top v_c)-1)\cdot v_c$$

We can actually simplify it a bit as $\sigma(x) = 1-\sigma(-x)$:

$$\frac{\partial{L}}{\partial{v_{n_i}}}= - (\sigma(-v_{n_i}^\top v_c)-1)\cdot v_c =  (1 - \sigma(-v_{n_i}^\top v_c))\cdot v_c = \sigma(v_{n_i}^\top v_c)\cdot v_c$$

We can change it to vectorised form (for all $i$ at the same time). Let's assume $V$ has shape (number_of_negatives, embedding_size):

$$\frac{\partial{L}}{\partial{V_{n}}}=\sigma(V_n v_c)\otimes v_c$$

#### Loss with respect to $v_c$

$$\frac{\partial{L}}{\partial{v_{c}}} = \frac{\partial{(- \log(\sigma(v_o^\top v_c))}}{\partial{v_c}} + \sum_i\frac{\partial{\log(\sigma(-v_{n_i}^\top v_c))}}{\partial{v_{c}}}$$

$$\frac{\partial{(- \log(\sigma(v_o^\top v_c))}}{\partial{v_c}} = -1 \cdot \frac{1}{\cancel{\sigma(v_o^\top v_c))}} \cdot \sigma(\cancel{v_o^\top v_c})(1-\sigma(v_o^\top v_c))\cdot v_o = (\sigma(v_o^\top v_c) - 1)\cdot v_o$$

$$\frac{\partial{\log(\sigma(-v_{n_i}^\top v_c))}}{\partial{v_{c}}}= - (\sigma(-v_{n_i}^\top v_c)-1)\cdot v_{n_i} = \sigma(v_{n_i}^\top v_c)\cdot v_{n_i}$$

We can again write the second part in vectorised form:

$$\sum_i\frac{\partial{\log(\sigma(-v_{n_i}^\top v_c))}}{\partial{v_{c}}}=\sigma(V_n v_c)^\top V_n$$

Putting it together, we get:

$$\frac{\partial{L}}{\partial{v_{c}}} = (\sigma(v_o^\top v_c) - 1)\cdot v_o + \sigma(V_n v_c)^\top V_n$$
