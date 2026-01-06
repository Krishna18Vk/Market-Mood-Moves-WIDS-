 """
NLP Pipeline for Market Mood Moves - Week 1-2 Implementation
Tokenization, Preprocessing, VADER, FinBERT
"""

import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import torch.nn.functional as F

# Download required NLTK data (run once)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('vader_lexicon', quiet=True)

def preprocess_text(text):
    """
    Week 1 §6 COMPLETE PIPELINE
    
    Steps (as per Week 1 reading):
    
    §6.1 TOKENIZATION: word_tokenize() splits into words/subwords
    - "Apple stock surges!" → ['apple', 'stock', 'surges', '!']
    
    §6.2 STOP WORDS: Remove common words (the, is, on, at)
    - stopwords.words('english') → 179 English stopwords
    - Keeps: ['apple', 'stock', 'surges']
    
    §6.3 LEMMATIZATION: Dictionary form (preserves meaning)
    - WordNetLemmatizer: 'running'→'run', 'studies'→'study'  
    - vs Stemming: 'running'→'runn' (chops)
    
    Financial text example:
    "Apple reports record earnings" → ['apple', 'report', 'record', 'earning']
    """
    
    # Clean text (remove special chars, lowercase)
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if w not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return tokens

def get_vader_score(text):
    """
    VADER sentiment wrapper - Week 1 §8.2
    Returns: {'neg':, 'neu':, 'pos':, 'compound':}
    """
    sia = SentimentIntensityAnalyzer()
    return sia.polarity_scores(text)

def get_finbert_score(text):
    """
    FinBERT sentiment wrapper - Week 1 §8.3, Week 2 §5
    Returns: [positive_prob, negative_prob, neutral_prob]
    """
    tokenizer = BertTokenizer.from_pretrained('ProsusAI/finbert')
    model = BertForSequenceClassification.from_pretrained('ProsusAI/finbert')
    
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)
    
    # FinBERT labels: [positive, negative, neutral]
    return probs[0].tolist()