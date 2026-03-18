import numpy as np
from datasets import load_dataset
from collections import Counter

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def nearest_neighbors(word, W_in, word2idx, idx2word, top_n=5):
    if word not in word2idx:
        print(f"'{word}' not in vocabulary")
        return
    
    idx = word2idx[word]
    v = W_in[idx]
    
    norms = np.linalg.norm(W_in, axis=1)
    sims = W_in @ v / (norms * np.linalg.norm(v))
    
    top_indices = np.argsort(sims)[::-1][1:top_n+1]
    for i in top_indices:
        print(f"  {idx2word[i]}: {sims[i]:.4f}")

def analogy(word_a, word_b, word_c, W_in, word2idx, idx2word, top_n=3):
    """word_a is to word_b as word_c is to ?"""
    for w in [word_a, word_b, word_c]:
        if w not in word2idx:
            print(f"'{w}' not in vocabulary")
            return
    
    v_a = W_in[word2idx[word_a]]
    v_b = W_in[word2idx[word_b]]
    v_c = W_in[word2idx[word_c]]
    
    target = v_b - v_a + v_c
    
    norms = np.linalg.norm(W_in, axis=1)
    sims = W_in @ target / (norms * np.linalg.norm(target))
    
    exclude = {word2idx[word_a], word2idx[word_b], word2idx[word_c]}
    top_indices = [i for i in np.argsort(sims)[::-1] if i not in exclude][:top_n]
    
    print(f"'{word_a}' is to '{word_b}' as '{word_c}' is to:")
    for i in top_indices:
        print(f"  {idx2word[i]}: {sims[i]:.4f}")

if __name__ == "__main__":
    W_in = np.load("W_in.npy")
    W_out = np.load("W_out.npy")

    ds = load_dataset("afmck/text8", split="train")
    corpus = "".join(ds["text"]).split()
    
    counter = Counter(corpus)
    vocabulary = [word for word, count in counter.items() if count > 5]
    i2w = {idx: word for idx, word in enumerate(vocabulary)}
    w2i = {word: idx for idx, word in i2w.items()}

    print("Nearest neighbors:")
    for word in ["king", "france", "capital", "simon"]:
        print(f"\n'{word}':")
        nearest_neighbors(word, W_in, w2i, i2w)

    print("\nAnalogies:")
    analogy("king", "queen", "man", W_in, w2i, i2w)
    analogy("paris", "france", "berlin", W_in, w2i, i2w)
    analogy("slow", "slower", "fast", W_in, w2i, i2w)