# Shakespearean Character-Level RNN

## Abstract
This project implements a generative Character-Level Recurrent Neural Network (RNN) designed to simulate the stylistic and syntactic patterns of Shakespearean English. Unlike large-scale Transformer models (e.g., GPT), this lightweight architecture demonstrates the capabilities of Gated Recurrent Units (GRU) in sequential data modeling, specifically focusing on next-character prediction tasks. The model achieves a low perplexity score, indicating a strong grasp of the dataset's vocabulary and structure.

## 1. Model Architecture
The model is built using TensorFlow/Keras and follows a sequential architecture optimized for time-series forecasting where the time steps are character indices.

| Layer Type | Parameters | Description |
| :--- | :--- | :--- |
| **Input** | `(None,)` | Accepts variable-length integer sequences. |
| **Embedding** | `Dim: 64` | Maps character indices to dense vectors of size 64. |
| **GRU Layer 1** | `Units: 128` | Gated Recurrent Unit with `return_sequences=True`. |
| **Dropout** | `Rate: 0.2` | Regularization to prevent overfitting. |
| **GRU Layer 2** | `Units: 128` | Second recurrent layer for deeper context capture. |
| **Dropout** | `Rate: 0.2` | Regularization. |
| **Dense (Output)**| `Units: Vocab` | Fully Connected layer with **Softmax** activation. |

* **Total Parameters:** ~150,000 (Lightweight implementation)
* **Optimizer:** Adam
* **Loss Function:** Sparse Categorical Crossentropy

## 2. Dataset & Preprocessing
* **Source:** Andrej Karpathy's "Tiny Shakespeare" dataset.
* **Vocabulary:** ~39 distinct characters (English alphabet + punctuation).
* **Tokenization:** Character-level mapping (int-to-char).
* **Windowing Strategy:**
    * The text is sliced into windows of `101` characters.
    * **Input ($X$):** Characters $t_{0}$ to $t_{100}$.
    * **Target ($Y$):** Characters $t_{1}$ to $t_{101}$ (Shifted by 1).
    * This allows the model to learn the probability $P(c_t | c_{t-1}, ..., c_{t-n})$.

## 3. Evaluation Metrics
The model was evaluated on a strictly held-out test set (last 10% of the corpus).

### Perplexity (PP)
Perplexity is the primary metric used to evaluate the model's uncertainty. It is defined as the exponentiation of the cross-entropy loss:

$$PP(p) = e^{-\sum p(x) \log q(x)} \approx e^{loss}$$

* **Result:** ~5.95
* **Interpretation:** On average, the model is uncertain between approximately 6 possible characters for any given prediction step. A random model would have a perplexity equal to the vocabulary size (~39).

### Top-k Accuracy
Given the creative nature of the task, Top-1 accuracy is less informative than Top-k.
* **Top-5 Accuracy:** >92%
* Indicates that the correct next character is present in the model's top 5 predictions over 92% of the time.

## 4. Inference & Sampling Strategy
The generation process utilizes **Stochastic Sampling** via Temperature Scaling rather than greedy decoding (`argmax`).

$$P_i = \frac{\exp(z_i / T)}{\sum \exp(z_j / T)}$$

Where:
* $z$ are the logits predicted by the model.
* $T$ is the **Temperature**.
    * $T < 1.0$: Sharpens the probability distribution (more conservative/repetitive).
    * $T > 1.0$: Flattens the distribution (more random/creative).

## 5. Experimental Inputs (Prompts)
To test the model's capabilities, the following prompt categories are recommended:

### A. The Monologue Starter
*Input:* `To be, or not`
*Expected Behavior:* The model should recognize the famous Hamlet soliloquy structure and attempt to complete it, or generate a philosophical musing.

### B. The Dialogue Trigger
*Input:* `ROMEO:`
*Expected Behavior:* The colon (`:`) acts as a strong signal for the model to switch to "dialogue mode", likely generating a response indented as a script.

### C. The Narrative Context
*Input:* `The King hath`
*Expected Behavior:* Usage of archaic grammar ("hath" instead of "has"). The model should predict a verb or an object fitting for a royal subject.

### D. The Creative/Abstract
*Input:* `O love, thy`
*Expected Behavior:* Testing the poetic capability. Common in sonnets, this input usually triggers metaphorical text related to beauty, death, or nature.

---

## Limitations
* **Context Window:** The RNN has a limited effective receptive field. It may lose track of the subject in long sentences.
* **Semantics:** The model mimics syntax (grammar) and style, but does not possess true semantic understanding or knowledge of the world. It cannot answer factual questions.