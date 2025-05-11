# Anglo-American Poem Generator: A Transformer-Based Language Model

This repository contains a PyTorch implementation of a character-level Transformer-based language model designed to generate text in the style of Anglo-American poetry. The model is trained on a consolidated dataset of Anglo-American poems.

## Project Overview

The goal of this project was to build and train a generative language model from scratch, capable of learning patterns from poetic text and producing novel verses. The implementation focuses on core Transformer architecture components, including:

*   Token and Positional Embeddings
*   Multi-Head Self-Attention Mechanisms
*   FeedForward Networks
*   Layer Normalization and Dropout
*   Custom training and text generation logic

## Model Architecture

The language model is a decoder-only Transformer, similar in principle to GPT models, but implemented at a character level.

*   **Input:** A sequence of character token indices.
*   **Embeddings:**
    *   `nn.Embedding` for character tokens.
    *   `nn.Embedding` for positional encodings, added to token embeddings.
*   **Transformer Blocks (`n_layer`):** Each block consists of:
    1.  Layer Normalization
    2.  Multi-Head Self-Attention (`MultiHeadAttention`):
        *   Composed of multiple `Head` modules.
        *   Each `Head` implements scaled dot-product attention with causal masking (using `torch.tril`) to prevent attending to future tokens.
        *   Includes a projection layer and dropout.
    3.  Residual Connection
    4.  Layer Normalization
    5.  FeedForward Network (`FeedForward`):
        *   Two linear layers with a ReLU activation in between.
        *   Includes dropout.
    6.  Residual Connection
*   **Output:**
    *   Final Layer Normalization.
    *   A Linear layer (`lm_head`) to project the Transformer's output to vocabulary size, producing logits for the next character prediction.
*   **Loss Function:** `torch.nn.functional.cross_entropy` is used during training.

### Key Hyperparameters (from notebook):
*   `vocab_size`: Determined by the unique characters in the dataset.
*   `n_embd`: 384 (Embedding dimension)
*   `n_head`: 8 (Number of attention heads)
*   `n_layer`: 8 (Number of Transformer blocks)
*   `block_size`: 256 (Context window / sequence length)
*   `dropout`: 0.3
*   `batch_size`: 64
*   `learning_rate`: 1e-4
*   `max_iters`: 10,000

## Dataset

The model is trained on a dataset of Anglo-American poems. The notebook loads this from `/kaggle/input/anglo-american-poems/all_data.txt`.
*   The text is tokenized at the character level.
*   The data is split into 90% for training and 10% for validation.

## Training

*   The model is trained using the AdamW optimizer.
*   A custom training loop (`train` function) handles batching (`get_batch`), forward/backward passes, and optimizer steps.
*   Validation loss is periodically calculated (`evaluate` and `calculate_loss` functions) to monitor progress.
*   The training progress from the notebook shows the model successfully learning and reducing both training and validation loss over 10,000 iterations.

## Text Generation

*   The `generate` method in the `LanguageModel` class performs auto-regressive text generation.
*   It takes an initial context (e.g., a single null token) and iteratively predicts the next character based on the current sequence, using multinomial sampling from the probability distribution output by the model.
*   The notebook demonstrates generating 1000 new characters of text.

## How to Use (Conceptual Steps)

1.  **Prerequisites:**
    *   Python 3.12
    *   PyTorch
2.  **Prepare Data:**
    *   Obtain a text dataset (e.g., `all_data.txt` containing poems).
    *   Place it in a location accessible by the script/notebook.
3.  **Configure Hyperparameters:** Adjust parameters like `batch_size`, `block_size`, `n_embd`, etc., at the beginning of the script or notebook.
4.  **Run Training:**
    *   Instantiate the `LanguageModel` and optimizer.
    *   Call the `train` function.
5.  **Generate Text:**
    *   After training, use the `model.generate()` method to produce new text.

## Example Output

*(You can find a sample of the generated output in "generated output.txt")*

## Future Work / Potential Improvements

*   Experiment with larger datasets or different genres of text.
*   Implement more sophisticated sampling strategies for generation (e.g., top-k, top-p/nucleus sampling, temperature scaling).
*   Explore sub-word tokenization (e.g., BPE, SentencePiece) instead of character-level for potentially richer semantic understanding and handling of out-of-vocabulary words.
*   Increase model size (embedding dimension, number of layers/heads) and context window for improved performance (requires more compute).
*   Implement learning rate scheduling.

---