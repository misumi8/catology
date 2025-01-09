from fileinput import filename
from pprint import pprint
from langdetect import detect, detect_langs
from nltk.tokenize import word_tokenize
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import wordnet
import random
from rodict import romanian_dict
from googletrans import Translator

nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
def read_file_or_keyboard(file = ""):
    if file:
        with open(file, "r", encoding="utf-8") as f:
            text = f.read()
            return text
    else:
        text = input()
        return text

def identify_lang(text):
    language = detect(text)
    probabilities = detect_langs(text)
    print(f"Language: {language}")
    print(f"Probabilities: {probabilities}")
    return language

def get_stylometric_info(text):
    tokens = word_tokenize(text)
    # for token in tokens:
    #     if not (re.match(r'^[\w\-]+$', token) and not token.startswith('-') and not token.endswith('-')):
    #         print(token)
    words = [token.lower() for token in tokens if re.match(r'^[\w\-]+$', token) and not token.startswith('-') and not token.endswith('-')]
    word_count = len(words)
    char_count_no_spaces = sum([len(word) for word in tokens])
    print(f"\nNumber of characters (including spaces): {len(text)}")
    print(f"Number of characters (without spaces): {char_count_no_spaces}")
    print(f"Number of words: {word_count}")
    print(f"Number of sentences: {len([token for token in tokens if token == '.'])}")
    word_frequency = {}
    for token in tokens:
        lowercase_token = token.lower()
        if lowercase_token not in word_frequency:
            word_frequency[lowercase_token] = 1
        else:
            word_frequency[lowercase_token] += 1
    print("Word frequency (sorted):")
    word_frequency = {k: v for k, v in sorted(word_frequency.items(), key=lambda item: item[1], reverse=True) if k in words}
    freq_percentages = {k: round((v / word_count) * 100, 2) for k, v in list(word_frequency.items())[:(15 if len(words) >= 15 else len(words))]}
    with open("frequencies_test.txt", 'w', encoding='utf-8') as f:
        f.write(text + "\n\n")
        for freq in word_frequency:
            if freq in words:
                frequency = str(freq) + ": " + str(word_frequency[freq]) + "(" + str(round((word_frequency[freq] / word_count) * 100, 2)) + "%)"
                print(f"\t{frequency}")
                f.write(f"{frequency}\n")
    plt.figure(figsize=(10, 6))
    bars = plt.bar(freq_percentages.keys(), freq_percentages.values(), color='skyblue')
    plt.bar_label(bars, fmt='%.2f%%', fontsize=10)
    plt.xlabel('Words', fontsize=12)
    plt.ylabel('Percentage', fontsize=12)
    plt.xticks(rotation=45)
    plt.subplots_adjust(left=0.055, right=0.971, top=0.95, bottom=0.145)
    plt.show()


translator = Translator()

def translate_word(word, src_lang, dest_lang):
    """Translate a word from src_lang to dest_lang, handling errors gracefully."""
    try:
        translated = translator.translate(word, src=src_lang, dest=dest_lang)
        return translated.text if translated and translated.text else word
    except Exception as e:
        print(f"Error translating '{word}': {e}")
        return word  

def is_pos_compatible(lemma, pos_tag):
    pos_map = {
        'JJ': 'a',  
        'VB': 'v',  
        'NN': 'n',  
        'RB': 'r', 
    }
    
    if pos_tag in pos_map:
        return pos_map[pos_tag] in lemma.synset().pos()
    return False

def get_replacements(word, pos_tag, lang):
    replacements = set()

    if lang == "en":
        for synset in wordnet.synsets(word):
            for lemma in synset.lemmas():
                if lemma.name() != word and is_pos_compatible(lemma, pos_tag):
                    replacements.add(lemma.name().replace('_', ' '))

            for hypernym in synset.hypernyms():
                for lemma in hypernym.lemmas():
                    if is_pos_compatible(lemma, pos_tag):
                        replacements.add(lemma.name().replace('_', ' '))

            for lemma in synset.lemmas():
                for antonym in lemma.antonyms():
                    if is_pos_compatible(antonym, pos_tag):
                        replacements.add("not " + antonym.name().replace('_', ' '))
    
    elif lang == "ro":
        if word in romanian_dict:
            replacements.update(romanian_dict[word]['synonyms'])
            
            for antonym in romanian_dict[word]['antonyms']:
                negated_antonym = "nu " + antonym
                replacements.add(negated_antonym)
            
            replacements.update(romanian_dict[word]['hypernyms'])

    else:
        translated_word = translate_word(word, src_lang=lang, dest_lang="en")
        print(f"Translated '{word}' ({lang}) to '{translated_word}' (en)")
        
        for synset in wordnet.synsets(translated_word):
            for lemma in synset.lemmas():
                if lemma.name() != translated_word and is_pos_compatible(lemma, pos_tag):
                    replacements.add(lemma.name().replace('_', ' '))

            for hypernym in synset.hypernyms():
                for lemma in hypernym.lemmas():
                    if is_pos_compatible(lemma, pos_tag):
                        replacements.add(lemma.name().replace('_', ' '))

            for lemma in synset.lemmas():
                for antonym in lemma.antonyms():
                    if is_pos_compatible(antonym, pos_tag):
                        replacements.add("not " + antonym.name().replace('_', ' '))

        replacements = {translate_word(replacement, src_lang="en", dest_lang=lang) for replacement in replacements}

    return replacements

def replace_words(text, replacement_fraction=0.2):
    words = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(words)
    num_to_replace = max(1, int(len(words) * replacement_fraction))
    lang = identify_lang(text)
    
    indices = list(range(len(words)))
    random.shuffle(indices)

    replaced = set()
    for idx in indices:
        word = words[idx]
        pos_tag = pos_tags[idx][1]
        
        if idx in replaced or not word.isalpha():
            continue
        
        replacements = get_replacements(word, pos_tag, lang)
        
        if replacements:
            replacement = random.choice(list(replacements))
            words[idx] = replacement
            replaced.add(idx)

        if len(replaced) >= num_to_replace:
            break

    return ' '.join(words)

def replace_words_in_file(input_file, output_file, replacement_fraction=0.2):
    original_text = read_file_or_keyboard(input_file)
    alternative_text = replace_words(original_text, replacement_fraction)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("Original text:\n")
        f.write(original_text + "\n\n")
        f.write("Alternative text:\n")
        f.write(alternative_text + "\n")

    print(f"Processed text saved to {output_file}")

text = read_file_or_keyboard("KR NLP/ro_test.txt")
#print(identify_lang(text))
#replace_words_in_file("KR NLP/ro_test.txt", "KR NLP/processed_text.txt", replacement_fraction=0.2)

#pprint(text)
get_stylometric_info(text)

