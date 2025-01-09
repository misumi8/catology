import nltk
from nltk.corpus import wordnet
import random

# Ensure you have the necessary NLTK data
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')

def get_replacements(word):
    """Get possible replacements for a word: synonyms, hypernyms, or negated antonyms."""
    replacements = set()
    
    # Get synonyms
    for synset in wordnet.synsets(word):
        for lemma in synset.lemmas():
            replacements.add(lemma.name().replace('_', ' '))
            
        # Get hypernyms
        for hypernym in synset.hypernyms():
            for lemma in hypernym.lemmas():
                replacements.add(lemma.name().replace('_', ' '))

        # Get negated antonyms
        for lemma in synset.lemmas():
            for antonym in lemma.antonyms():
                replacements.add("not " + antonym.name().replace('_', ' '))

    return replacements

def replace_words(text, replacement_fraction=0.2):
    """Replace at least a fraction of the words in the text."""
    words = nltk.word_tokenize(text)
    num_to_replace = max(1, int(len(words) * replacement_fraction))

    # Shuffle indices to pick random words to replace
    indices = list(range(len(words)))
    random.shuffle(indices)

    replaced = set()  # Keep track of indices already replaced
    for idx in indices:
        word = words[idx]
        
        # Skip if the word has already been replaced or is not alphanumeric
        if idx in replaced or not word.isalpha():
            continue

        replacements = get_replacements(word)
        
        if replacements:
            replacement = random.choice(list(replacements))
            words[idx] = replacement
            replaced.add(idx)

        # Stop if we've replaced enough words
        if len(replaced) >= num_to_replace:
            break

    return ' '.join(words)

# Example usage
original_text = "The house is far from the city but close to a beautiful forest."
print("Original text:", original_text)

alternative_text = replace_words(original_text, replacement_fraction=0.2)
print("Alternative text:", alternative_text)
