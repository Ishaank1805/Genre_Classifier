# Linguistic Feature Extractor

A modular Python library for extracting linguistic features from text data for genre classification and text analysis.

## Project Structure
```
linguistic_features/
├── base_extractor.py              # Base class for all extractors
├── linguistic_features.py         # Main feature extractor
└── features/
    ├── basic_features.py          # Basic text statistics
    ├── char_diversity_features.py # Character-level diversity
    ├── pos_features.py            # Part-of-speech ratios
    ├── syntax_features.py         # Syntactic complexity
    ├── dialogue_features.py       # Dialogue detection
    ├── pronoun_features.py        # Pronoun usage
    ├── temporal_features.py       # Tense and time
    ├── narrative_features.py      # Narrative elements
    ├── academic_features.py       # Academic markers
    ├── punctuation_features.py    # Punctuation patterns
    ├── discourse_features.py      # Discourse markers
    └── readability_features.py    # Readability scores
```

## Installation

### Required Dependencies
```bash
pip install numpy pandas stanza spacy scikit-learn
python -m spacy download en_core_web_sm
```

### First-time Setup
```python
import stanza
stanza.download('en')  # Run once to download English models
```

## How to Run the Code

### Step 1: Set up the project structure
Create the following directory structure:
```
your_project/
├── base_extractor.py
├── linguistic_features.py
└── features/
    ├── __init__.py  # Create empty file
    ├── basic_features.py
    ├── char_diversity_features.py
    ├── pos_features.py
    ├── syntax_features.py
    ├── dialogue_features.py
    ├── pronoun_features.py
    ├── temporal_features.py
    ├── narrative_features.py
    ├── academic_features.py
    ├── punctuation_features.py
    ├── discourse_features.py
    └── readability_features.py
```

### Step 2: Create your script
Create a new Python file (e.g., `extract_my_features.py`):

```python
# extract_my_features.py
from linguistic_features import FeatureExtractor
import pandas as pd

# Load your CSV data
df = pd.read_csv('your_data.csv')  # Replace with your CSV filename

# Option 1: Extract all features
extractor = FeatureExtractor(feature_groups='all')

# Option 2: Extract specific feature groups only
# extractor = FeatureExtractor(feature_groups=['basic', 'dialogue', 'pronouns'])

# Extract features from your data
features = extractor.extract_features_from_dataframe(
    df, 
    text_column='text',      # Replace 'text' with your column name
    label_column='label',    # Optional: Replace 'label' with your label column
    cache_file='features_cache.csv'  # Optional: Cache results
)

# Save the extracted features
features.to_csv('extracted_features.csv', index=False)
print(f"Features saved to extracted_features.csv")
```

### Step 3: Run the script
```bash
python extract_my_features.py
```

## Complete Working Example

### Example 1: Basic Usage
```python
from linguistic_features import FeatureExtractor
import pandas as pd

# Create sample data
data = {
    'text': [
        "Hello! How are you today? I'm feeling great.",
        "The research indicates that climate change affects biodiversity.",
        "She walked slowly to the door, her heart pounding with fear."
    ],
    'genre': ['dialogue', 'academic', 'narrative']
}
df = pd.DataFrame(data)

# Extract basic features only
extractor = FeatureExtractor(feature_groups=['basic'])
features = extractor.extract_features_from_dataframe(df, text_column='text')
print(features)
```

### Example 2: Multiple Feature Groups
```python
# Extract dialogue and pronoun features for fiction analysis
extractor = FeatureExtractor(feature_groups=['dialogue', 'pronouns', 'narrative'])
features = extractor.extract_features_from_dataframe(
    df, 
    text_column='text',
    label_column='genre'
)

# View specific features
print("Dialogue features:")
print(features[['dialogue_count', 'dialogue_ratio', 'speech_verb_count']])
```

### Example 3: Processing Large Dataset with Progress
```python
# For large datasets
df = pd.read_csv('large_dataset.csv')

# Extract features with caching
extractor = FeatureExtractor(feature_groups=['basic', 'pos', 'readability'])
features = extractor.extract_features_from_dataframe(
    df,
    text_column='content',  # Your text column
    cache_file='large_dataset_features.csv'  # Will load from cache if exists
)
```

## Available Feature Groups

### 1. Basic Features (`'basic'`)
- Average sentence length
- Standard deviation of sentence length
- Lexical density

### 2. Character Diversity (`'char_diversity'`)
- TTR (Type-Token Ratio)
- MAAS TTR
- MSTTR (Mean Segmental TTR)
- MATTR (Moving Average TTR)
- MTLD (Measure of Textual Lexical Diversity)
- MTLD-MA (Moving Average MTLD)
- Yule's K
- VocD

### 3. POS Features (`'pos'`)
- POS tag ratios (adverb/noun, adjective/verb, etc.)
- Content vs function word ratios

### 4. Syntactic Features (`'syntax'`)
- Dependency relations
- Syntactic complexity measures
- Tree depth statistics
- Dependency distances

### 5. Dialogue Features (`'dialogue'`)
- Direct speech count and ratio
- Speech verb usage
- Speech tag patterns
- Dialogue diversity

### 6. Pronoun Features (`'pronouns'`)
- First/second/third person pronouns
- Singular/plural distinctions
- Person ratio comparisons

### 7. Temporal Features (`'temporal'`)
- Past/present tense ratios
- Time adverbs
- Temporal markers

### 8. Narrative Features (`'narrative'`)
- Action verbs
- Sensory words
- Emotion words
- Character names
- Place indicators

### 9. Academic Features (`'academic'`)
- Citation patterns
- Numbers and statistics
- Technical terms
- Parentheticals

### 10. Punctuation Features (`'punctuation'`)
- Exclamation/question marks
- Ellipses
- Comma density
- Semicolons and colons

### 11. Discourse Features (`'discourse'`)
- Causal markers
- Contrast markers
- Addition markers

### 12. Readability Features (`'readability'`)
- Average words per sentence
- Syllables per word
- Flesch Reading Ease
- Gunning Fog Index

## Customization

### Select Specific Feature Groups
```python
# Use only specific feature groups
extractor = FeatureExtractor(feature_groups=['basic', 'dialogue', 'pronouns'])

# Or use all features
extractor = FeatureExtractor(feature_groups='all')
```

### Extract Individual Features
```python
# Extract features from a single text
text = "Your text here..."
features = extractor.extract_features(text)
```

### Process Large Datasets with Caching
```python
features = extractor.extract_features_from_dataframe(
    df,
    text_column='content',
    cache_file='features_cache.csv'  # Cache results to avoid re-processing
)
```

## Common Use Cases

### 1. Genre Classification
```python
from linguistic_features import FeatureExtractor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pandas as pd

# Load data
df = pd.read_csv('genre_data.csv')

# Extract features for genre classification
extractor = FeatureExtractor(feature_groups=[
    'dialogue', 'pronouns', 'narrative', 'academic', 'temporal'
])
X = extractor.extract_features_from_dataframe(df, text_column='text')
y = df['genre'].values

# Train classifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train_scaled, y_train)
print(f"Accuracy: {model.score(X_test_scaled, y_test):.2f}")
```

### 2. Single Text Analysis
```python
# Analyze a single text
text = """
"I can't believe you did that!" she exclaimed, tears streaming down her face.
John looked away, unable to meet her gaze. The silence between them grew heavy.
"""

extractor = FeatureExtractor(feature_groups=['dialogue', 'emotion', 'narrative'])
features = extractor.extract_features(text)

print("Dialogue count:", features['dialogue_count'])
print("Emotion words:", features['emotion_word_count'])
print("Action verbs:", features['action_verb_count'])
```

### 3. Batch Processing Large Files
```python
# Process large CSV in batches
def process_large_csv(filename, batch_size=1000):
    extractor = FeatureExtractor(feature_groups=['basic', 'readability'])
    
    all_features = []
    for chunk in pd.read_csv(filename, chunksize=batch_size):
        features = extractor.extract_features_from_dataframe(
            chunk, 
            text_column='text'
        )
        all_features.append(features)
    
    return pd.concat(all_features, ignore_index=True)

# Process file
features = process_large_csv('large_corpus.csv')
features.to_csv('corpus_features.csv', index=False)
```

## Feature Output

The extractor returns a pandas DataFrame where:
- Each row corresponds to a document
- Each column is a linguistic feature
- All features are numerical (counts, ratios, or scores)

## Advanced Usage

### Custom Feature Selection
```python
# Get feature names for a specific group
basic_features = extractor.get_feature_names('basic')
print(basic_features)

# Extract and select specific features
all_features = extractor.extract_features_from_dataframe(df, text_column='text')
selected_features = all_features[['avg_sen_len', 'dialogue_count', 'pronoun_first_singular_ratio']]
```

### Integration with Machine Learning
```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Extract features
X = extractor.extract_features_from_dataframe(df, text_column='text')
y = df['label'].values

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)
```

## Adding Custom Feature Groups

To add your own feature group:

1. Create a new file in the `features/` directory:
```python
# features/custom_features.py
from base_extractor import BaseFeatureExtractor

class CustomFeatureExtractor(BaseFeatureExtractor):
    def extract(self, text):
        features = {}
        # Add your feature extraction logic
        features['custom_feature'] = len(text)
        return features
```

2. Import and register in `linguistic_features.py`:
```python
from features.custom_features import CustomFeatureExtractor

# In the __init__ method, add to available_groups:
self.available_groups = {
    # ... existing groups ...
    'custom': CustomFeatureExtractor
}
```

3. Use your custom features:
```python
extractor = FeatureExtractor(feature_groups=['custom'])
```