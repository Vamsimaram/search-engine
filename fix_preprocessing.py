import re

def simplified_preprocess_text(text):
    """Simplified preprocessing: lowercase, remove special chars, simple tokenization"""
    # Convert to lowercase and remove special characters
    text = re.sub(r'[^\w\s]', '', text.lower())
    
    # Simple tokenization by splitting on whitespace
    tokens = text.split()
    
    # Return as joined tokens
    return ' '.join(tokens)

# Update the preprocessing.py file
with open('src/preprocessing.py', 'r') as file:
    content = file.read()

# Replace the complex tokenization with a simpler approach
modified_content = content.replace(
    'def preprocess_text(text):\n    """Basic preprocessing: lowercase, remove special chars, tokenize, remove stopwords, stem"""\n    # Convert to lowercase and remove special characters\n    text = re.sub(r\'[^\\w\\s]\', \'\', text.lower())\n    \n    # Tokenize\n    tokens = word_tokenize(text)\n    \n    # Remove stopwords and stem\n    stop_words = set(stopwords.words(\'english\'))\n    stemmer = PorterStemmer()\n    \n    processed_tokens = [stemmer.stem(token) for token in tokens if token not in stop_words]\n    \n    return \' \'.join(processed_tokens)',
    
    'def preprocess_text(text):\n    """Simplified preprocessing: lowercase, remove special chars, simple tokenization"""\n    # Convert to lowercase and remove special characters\n    text = re.sub(r\'[^\\w\\s]\', \'\', text.lower())\n    \n    # Simple tokenization by splitting on whitespace\n    tokens = text.split()\n    \n    # Return as joined tokens - no stemming or stopword removal for now\n    return \' \'.join(tokens)'
)

# Write the modified content back
with open('src/preprocessing.py', 'w') as file:
    file.write(modified_content)

print("Updated preprocessing.py with simplified tokenization approach.")