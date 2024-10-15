import numpy as np
import torch
import torch.nn.functional as F


def train(
    data: str,
    emb_size: int = 64,
    window_size: int = 7,
    epochs: int = 30,
    lr: float = 1e-3,
) -> dict[str, np.ndarray]:
    """
    Trains a simple Word2Vec-like word embedding model using a context window and negative sampling.

    Args:
        data (str): Cleaned document without punctuation, with words separated by spaces.
        emb_size (int, optional): Size of the word embeddings. Defaults to 64.
        window_size (int, optional): Size of the context window. Defaults to 7.
        epochs (int, optional): Number of training epochs. Defaults to 30.
        lr (float, optional): Learning rate for the optimizer. Defaults to 1e-3.

    Returns:
        dict[str, np.ndarray]: A dictionary where keys are words from the vocabulary and
                               values are numpy arrays representing the word embeddings.
    """
    # Tokenize the input data
    words = data.split()
    vocab = sorted(set(words))
    vocab_size = len(vocab)

    # Create word-to-index mapping
    wtoi = {w: i for i, w in enumerate(vocab)}

    # Convert words to tensor of word indices
    ids = torch.tensor([wtoi[w] for w in words], dtype=torch.long)

    # Initialize word embeddings (center and context embeddings)
    emb_o = torch.randn((vocab_size, emb_size), requires_grad=True)  # Center embeddings
    emb_c = torch.randn(
        (vocab_size, emb_size), requires_grad=True
    )  # Context embeddings

    # Define optimizer
    optimizer = torch.optim.Adam([emb_o, emb_c], lr=lr)

    # Training loop
    for _ in range(epochs):
        for pos_c, ix_c in enumerate(ids):
            # Determine the context word indices for the current center word
            context_indices = [
                ids[pos_o]
                for pos_o in range(
                    max(0, pos_c - window_size), min(len(ids), pos_c + window_size + 1)
                )
                if pos_o != pos_c
            ]
            ix_o = torch.tensor(context_indices, dtype=torch.long)

            # Compute similarity scores between the center word and all other words
            similarities = emb_o[ix_c] @ emb_c.T

            # Compute loss using cross-entropy between the true context words and predicted similarities
            loss = F.cross_entropy(similarities.unsqueeze(0), ix_o.unsqueeze(0))

            # Backpropagation and optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Average the center and context embeddings to create final word vectors
    emb_avg = (emb_o.detach() + emb_c.detach()) / 2

    # Create dictionary mapping words to their corresponding embeddings
    w2v_dict = {w: emb_avg[wtoi[w]].numpy() for w in vocab}

    return w2v_dict
