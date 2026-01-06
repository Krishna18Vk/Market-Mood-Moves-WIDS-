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
