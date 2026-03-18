import numpy as np
from datasets import load_dataset
from collections import Counter
from argparse import ArgumentParser


def load_text8(split="train"):
    ds = load_dataset("afmck/text8", split=split)
    return "".join(ds["text"])

def generate_training_pairs(encoded, max_window):
    for i, center in enumerate(encoded):
        window = np.random.randint(1, max_window + 1)
        start = max(0, i - window)
        end = min(len(encoded), i + window + 1)
        for j in range(start, end):
            if j != i:
                yield center, encoded[j]

def build_noise_distribution(counter, w2i, power):
    vocab_size = len(w2i)
    noise = np.zeros(vocab_size)
    
    for word, idx in w2i.items():
        noise[idx] = counter[word] ** power
    
    noise /= noise.sum()
    return noise

def sample_negatives(noise_dist, k):
    return np.random.choice(len(noise_dist), size=k, p=noise_dist)

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def training_step(center, context, negatives, W_in, W_out, lr):
    v_c = W_in[center] #embed_dim
    v_o = W_out[context] #embed_dim
    v_n = W_out[negatives] #k, embed_dim

    pos_score = sigmoid(v_o @ v_c)
    neg_scores = sigmoid(v_n @ v_c)

    loss = -np.log(pos_score) - np.sum(np.log(1 - neg_scores))

    grad_v_o = (pos_score - 1) * v_c
    grad_v_n = neg_scores[:, None] * v_c
    grad_v_c = (pos_score - 1) * v_o + neg_scores @ v_n

    W_out[context] -= lr * grad_v_o
    np.add.at(W_out, negatives, -lr * grad_v_n)
    W_in[center] -= lr * grad_v_c

    return loss

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def nearest_neighbors(word, W_in, word2idx, idx2word, top_n=5):
    if word not in word2idx:
        print(f"'{word}' not in vocabulary")
        return
    
    idx = word2idx[word]
    v = W_in[idx]
    
    # cosine similarity
    norms = np.linalg.norm(W_in, axis=1)
    sims = W_in @ v / (norms * np.linalg.norm(v))
    
    top_indices = np.argsort(sims)[::-1][1:top_n+1]
    
    for i in top_indices:
        print(f"  {idx2word[i]}: {sims[i]:.4f}")

def train(encoded, W_in, W_out, noise_dist, lr=0.025, max_window=5, k=5, n_epochs=1):
    for epoch in range(n_epochs):
        total_loss = 0
        n_steps = 0
        
        for center, context in generate_training_pairs(encoded, max_window):
            negatives = np.random.choice(len(noise_dist), size=k, p=noise_dist)
            
            loss = training_step(center, context, negatives, W_in, W_out, lr)
            
            total_loss += loss
            n_steps += 1
            
            if n_steps % 10_000 == 0:
                print(f"Epoch {epoch+1} | Step {n_steps} | Avg loss: {total_loss / n_steps:.4f}")
        
        print(f"Epoch {epoch+1} complete | Avg loss: {total_loss / n_steps:.4f}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--frequency_cutoff", type=int, default=5)
    parser.add_argument("--max_window", type=int, default=5)
    parser.add_argument("--unigram_power", type=float, default=0.75)
    parser.add_argument("--negatives", type=int, default=10)
    parser.add_argument("--embed_size", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()

    corpus = load_text8()
    print(f"Loaded text8 corpus with {len(corpus)} characters.")
    corpus = corpus.split(sep=' ')
    counter = Counter(corpus)

    print(f"\nTop 5 most common words and their number of occurences:")
    print(counter.most_common(5))

    vocabulary = [word for word, count in counter.items() if count > args.frequency_cutoff]
    i2w = {idx: word for idx, word in enumerate(vocabulary)}
    w2i = {word: idx for idx, word in i2w.items()}

    encoded = [w2i[token] for token in corpus if token in w2i]

    print(f"\nVocabulary size: {len(w2i)}")
    print(f"Encoded corpus length: {len(encoded)}\n")

    noise = build_noise_distribution(counter, w2i, args.unigram_power)

    W_in = np.random.randn(len(w2i), args.embed_size) * 0.01
    W_out = np.random.randn(len(w2i), args.embed_size) * 0.01

    train(
        encoded=encoded,
        W_in=W_in,
        W_out=W_out,
        noise_dist=noise,
        lr=args.lr,
        max_window=args.max_window,
        k=args.negatives,
        n_epochs=args.epochs
    )

    # eval
    print("\nNearest neighbors:")
    for word in ["king", "france", "computer", "run"]:
        print(f"\n'{word}':")
        nearest_neighbors(word, W_in, w2i, i2w)

    # save embeddings
    np.save("W_in.npy", W_in)
    np.save("W_out.npy", W_out)
    print("\nEmbeddings saved.")