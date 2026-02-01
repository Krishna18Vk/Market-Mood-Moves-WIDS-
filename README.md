
 # Market Mood Moves: Sentiment-Driven Stock Prediction
**WiDS 5.0 Weeks 1-2 Midterm Submission | Ameya Bansal | IIT Bombay DESE**

## 🎓 What I Learnt from Week 1 Reading Material

### **Week 1 Reading Guide Coverage**
**1. Pandas & NumPy for Financial Data (Sections 3-4)**
- Loading OHLC stock data, calculating daily returns, handling time series
- Implemented in `03_returns_stationarity.ipynb`

**2. NLP Text Preprocessing Pipeline (Section 6)**  
- **Tokenization**: `word_tokenize()` splits sentences into words
- **Stop Words Removal**: Removed 179 English stopwords that don't affect sentiment  
- **Lemmatization**: `WordNetLemmatizer()` converts "running" → "run" (preserves meaning)
- Full pipeline implemented in `src/nlp_pipeline.py` + demo in `01_data_preprocessing.ipynb`

**3. Data Collection APIs (Section 7)**
- **NewsAPI**: Fetch business headlines (100 requests/day free tier)
- **yfinance**: Download AAPL OHLC data for return calculations

**4. Sentiment Analysis (Section 8)**
- **VADER**: Lexicon-based scoring for quick sentiment (-1 to +1 compound score)
- **FinBERT**: Transformer model fine-tuned for financial text (`ProsusAI/finbert`)
- Both implemented in `02_sentiment_analysis.ipynb`

**5. Quantitative Finance Basics (Sections 10-11)**
- Performance metrics: Sharpe ratio, Maximum Drawdown, CAGR
- ADF stationarity test (Challenge §11.3) - raw prices non-stationary, returns stationary

### **Week 2 Reading Material - BERT Deep Dive**
**1. Embeddings Evolution**
- Static embeddings (Word2Vec) fail on polysemy ("bank" = river OR finance)
- BERT uses **contextual embeddings** - same word, different vectors by context

**2. BERT Technical Architecture**
- **Input**: Token embeddings (WordPiece) + Positional + Segment embeddings
- **Self-Attention**: `Attention(Q,K,V) = softmax(QK^T/√d_k)V`
- **Pre-training**: Masked Language Modeling (MLM 80-10-10 rule) + Next Sentence Prediction

**3. FinBERT Adaptation Pipeline**
BERT (Wikipedia trained)
↓ Further pre-training: TRC2-Financial (29M financial words)
↓ Fine-tuning: Financial PhraseBank → 97% accuracy

text

## 💻 Code Implementation - What I Built

### **`src/nlp_pipeline.py` - My Production Functions**
```python
def preprocess_text(text):
    """Week 1 §6.1-6.3: Complete tokenization → stopwords → lemmatization pipeline"""
    # My implementation matching PDF examples exactly

def get_vader_score(text): 
    """Week 1 §8.2: Returns {'neg':, 'neu':, 'pos':, 'compound':}"""

def get_finbert_score(text):
    """Week 1 §8.3 + Week 2 §5: Returns [positive, negative, neutral] probabilities"""
notebooks/01_data_preprocessing.ipynb - My NLP Pipeline Demo
text
What I did:
1. Created sample financial headlines dataset
2. Applied my preprocess_text() function to each
3. Showed token count reduction (raw → processed)
4. Verified pipeline works end-to-end
notebooks/02_sentiment_analysis.ipynb - My Sentiment Pipeline
text
What I implemented:
1. Raw headline → preprocess_text() → clean tokens  
2. Dual scoring: get_vader_score() + get_finbert_score()
3. Comparison table: VADER vs FinBERT on financial text
4. Verified FinBERT gives [positive, negative, neutral] probabilities
notebooks/03_returns_stationarity.ipynb - My Quant Analysis
text
What I built:
1. yfinance AAPL data download (Week 1 §7.2)
2. Daily returns calculation: (P_t - P_{t-1})/P_{t-1}
3. ADF stationarity test (Week 1 §11.3 Challenge)
4. Results: Raw prices NON-stationary → Returns STATIONARY ✓
📋 Repository Structure
text
.
├── README.md                    # This explanation
├── requirements.txt             # pip install dependencies
├── src/
│   └── nlp_pipeline.py         # My 3 core functions
└── notebooks/
    ├── 01_data_preprocessing.ipynb     # My NLP demo
    ├── 02_sentiment_analysis.ipynb     # My sentiment demo  
    └── 03_returns_stationarity.ipynb   # My quant demo
🚀 How to Run My Code
bash
pip install -r requirements.txt
jupyter notebook notebooks/
Recommended order:

01_data_preprocessing.ipynb (test my NLP pipeline)

02_sentiment_analysis.ipynb (test sentiment scoring)

03_returns_stationarity.ipynb (test quant analysis)

✅ Syllabus Coverage Confirmation
text
Week 1 Reading Guide:
✅ Sections 3-4: Pandas/NumPy financial analysis ✓
✅ Section 6: Complete NLP preprocessing pipeline ✓  
✅ Section 7: API data collection implemented ✓
✅ Section 8: VADER + FinBERT sentiment ✓
✅ Section 11.3: ADF stationarity challenge ✓

Week 2 Reading Guide: 
✅ BERT architecture understanding ✓
✅ FinBERT domain adaptation pipeline ✓
✅ Embeddings evolution ✓
=======
# Market-Mood-Moves-WIDS-

--
# 📈 Market Mood & Moves: Sentiment-Driven Stock Prediction

### A Multimodal Deep Learning Framework for Financial Forecasting

**Mentors:** Meet & Sarthak | **Mentee:** [Your Name] | **Project:** WiDS 5.0

---

## 📖 Project Overview

**Market Mood & Moves** is a quantitative finance project that challenges the traditional reliance on purely numerical data for stock prediction. By fusing **unstructured news sentiment** (from FinBERT) with **structured technical indicators**, this project builds a "Multimodal" prediction engine.

The core hypothesis is that **Market Sentiment** acts as a leading indicator for price volatility. The model uses an **Attention-based LSTM** (Long Short-Term Memory) network to learn which days in a 50-day window are most critical for predicting tomorrow's price direction.

## 🚀 Key Features

* **Deep Semantic Analysis:** Uses `ProsusAI/finbert` (a BERT model fine-tuned on financial text) to score news headlines.
* **Smart Tokenization:** Handles long financial reports using a **Chunking Strategy** to bypass the 512-token BERT limit.
* **Multimodal Fusion:** Combines Sentiment, RSI, MACD, Volatility, and Volume into a single feature vector.
* **Attention Mechanism:** A custom neural network layer that learns to "pay attention" to specific high-impact days (e.g., earnings calls) while ignoring noise.
* **Risk-Managed Strategy:** A "Signal Agreement" trading logic that only executes trades when both the LSTM (Price) and FinBERT (Sentiment) agree on the direction.

---

## 🛠️ Technical Workflow

### 1. Feature Engineering & Preprocessing

The raw data is transformed into a rich feature set:

* **Log Returns:**  are used instead of simple prices for statistical stationarity.
* **Technical Indicators:**
* **RSI (Relative Strength Index):** Measures overbought/oversold conditions.
* **MACD (Moving Average Convergence Divergence):** Captures momentum shifts.
* **Rolling Volatility:** Standard deviation of returns over a 20-day window.


* **Hybrid Scaling:** Returns are **Standardized** (Z-Score), while technical indicators are **Min-Max Scaled** to ensure stable gradients during training.

### 2. The NLP Pipeline (FinBERT)

We process financial news using the `ProsusAI/finbert` model.

* **Token Counting:** We first check for "long" documents that exceed 512 tokens.
* **Chunking Logic:** Long articles are split into 510-token chunks with overlap.
* **Inference:** Each chunk is scored independently, and the final sentiment is the average probability of the chunks. This ensures no critical information at the end of a long report is lost.

### 3. The Model Architecture (Attention-LSTM)

The model treats the stock market as a sequence problem.

* **Input:** A sequence of 50 trading days (Batch, 50, 6 Features).
* **LSTM Layers:** Two stacked LSTM layers capture temporal dependencies.
* **Attention Layer:** A custom linear layer calculates an "importance weight" () for every day in the sequence.
* *Intuition:* If Day 45 had a massive news crash, the model assigns it a high weight, ensuring it dominates the final prediction.



### 4. Trading Strategy (The Decision Engine)

The model outputs a predicted return for . We don't trade blindly on this number. We use a **Confirmation Filter**:

```python
def get_trade_signal(lstm_prediction, sentiment_score):
    if lstm_prediction > threshold AND sentiment_score > positive_threshold:
        return 'BUY'  # High Conviction
    elif lstm_prediction < -threshold AND sentiment_score < negative_threshold:
        return 'SELL' # High Conviction
    else:
        return 'HOLD' # Disagreement = Risk

```

---

## 📊 Visualizations Included

The code generates several key plots to validate performance:

1. **One-Step-Ahead Forecast:** Compares the model's "Next Day" prediction against the actual closing price.
2. **Strategy Signal Map:** A price chart overlaid with **Green Triangles (Buy)** and **Red Triangles (Sell)**, showing exactly where the algorithm would have entered the market.

## 📦 Requirements

To run this code, you will need the following libraries:

```txt
torch
transformers
pandas
numpy
matplotlib
scikit-learn

```

## 🏃 How to Run

1. **Prepare Data:** Ensure your dataframe `df` contains `Close`, `Volume`, and `News` columns.
2. **Run NLP:** Execute the FinBERT cells to generate the `Sentiment Score` column.
3. **Train:** Run the training loop. The loss should decrease over 100 epochs.
4. **Visualize:** Use the plotting cells to see the 'Buy/Sell' signals generated on the test set.

---

*Submitted as part of the WiDS 5.0 evaluation.*
