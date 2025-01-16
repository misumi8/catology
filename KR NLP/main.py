import os
from pprint import pprint
import json
import pandas as pd
from langdetect import detect, detect_langs
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import re
import nltk
import matplotlib.pyplot as plt
from pandas import Series
from rake_nltk import Rake
import google.generativeai as genai
from nltk.corpus import wordnet
import random
import copy

from sklearn.preprocessing import LabelEncoder

import backpropagation
from rodict import romanian_dict
from googletrans import Translator

nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('stopwords')

def read_file_or_keyboard(file=""):
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
                frequency = str(freq) + ": " + str(word_frequency[freq]) + " (" + str(round((word_frequency[freq] / word_count) * 100, 2)) + "%)"
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

# text = read_file_or_keyboard("KR NLP/ro_test.txt")
# #print(identify_lang(text))
# replace_words_in_file("KR NLP/ro_test.txt", "KR NLP/processed_text.txt", replacement_fraction=0.2)

#pprint(text)
language_map = {
    'ro': 'romanian',
    'en': 'english',
    'fr': 'french',
    'es': 'spanish',
}

def extract_keywords(text, detected_language):
    if detected_language not in language_map:
        print(f"Stop-words for language {detected_language} are not available.")
        return

    stop_words = set(stopwords.words(language_map[detected_language]))

    rake = Rake(max_length=1)
    rake.extract_keywords_from_text(text)
    raw_keywords = rake.get_ranked_phrases()

    # Filtering keywords that are stop-words
    # Stop words - commonly used words that don't offer too much useful information
    keywords = [
        phrase for phrase in raw_keywords
        if not any(word.lower() in stop_words for word in phrase.split())
    ][:10]

    print("\nExtracted Keywords:")
    pprint(keywords)

    # Sentences that contain the keywords
    sentences = sent_tokenize(text)
    keyword_sentences = {}

    for keyword in keywords:
        for sentence in sentences:
            if keyword in sentence and keyword not in keyword_sentences:
                keyword_sentences[keyword] = sentence
                break

    return keyword_sentences

def generate_sentences(keyword_sentences, detected_language):
    with open("A://gemini_api_key.txt", "r") as f:
        API_KEY = f.read()
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')

    for keyword, sentence in keyword_sentences.items():
        response = model.generate_content("Generate a single sentence in " + language_map[detected_language] + " that includes the following keywords: \"" + keyword +
                                          "\", ensuring that the keywords have the same meaning as they have in this sentence: \"" + sentence +
                                          "\". ")
        print(f"Keyword: {keyword}\nOriginal sentence: {sentence}\nGenerated sentence: {response.text}")

    # print("Generate a single sentence in " + language_map[
    #     detected_language] + " that includes the following keywords: \"" + keyword +
    # "\", ensuring that the keywords have the same meaning as they have in this sentence: \"" + sentence +
    # "\". ")

def generate_breed_description(breed):
    with open("A://gemini_api_key.txt", "r") as f:
        API_KEY = f.read()
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content("Write a maximum 5-sentence description of the " + breed + "cat breed, highlighting "
                                      "its key characteristics, temperament, and unique features. Be concise, engaging, and accurate.")
    return response.text

def generate_breed_comparison(breed_1, breed_2):
    with open("A://gemini_api_key.txt", "r") as f:
        API_KEY = f.read()
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content("Write a maximum 5-sentence comparison between the " + breed_1 + " and the " + breed_2 + " cat breeds, "
                                      "highlighting their key characteristics, temperament, and unique features. Be concise, engaging, and accurate.")
    return response.text

def init_input(file_name):
    df = pd.read_excel(file_name, engine='openpyxl')
    columns = df.columns.drop(["Breed"])
    label_encoder = LabelEncoder()
    df["Breed"] = label_encoder.fit_transform(df["Breed"]) + 1
    return {column : df[column].mean() for column in columns}, label_encoder

def analyze_description(columns, tokens, pos_tags):
    # pos_tags = {k: v for k, v in nltk.pos_tag(keywords)}
    # print(pos_tags)
    attributes = {}

    token_relations = {}
    for token in tokens:
        token_relations[token] = {"synonyms": set(), "antonyms": set(), "hypernyms": set()}
        for synset in wordnet.synsets(token):
            for lemma in synset.lemmas():
                # Synonyms
                if lemma.name() != token:
                    token_relations[token]["synonyms"].add(lemma.name().replace('_', ' '))
                # Antonyms
                if lemma.antonyms():
                    token_relations[token]["antonyms"].add(lemma.antonyms()[0].name().replace('_', ' '))
            # Hypernyms
            for hypernym in synset.hypernyms():
                token_relations[token]["hypernyms"].update(
                    [h.replace('_', ' ') for h in hypernym.lemma_names()]
                )

    for column in columns:
        for i, token in enumerate(tokens):
            synonyms = token_relations[token]["synonyms"]
            antonyms = token_relations[token]["antonyms"]
            hypernyms = token_relations[token]["hypernyms"]
            # print("----------------------")
            # print(synonyms)
            # print(antonyms)
            # print(hypernyms)
            # print("----------------------")
            if column.lower() in synonyms or column in hypernyms or column.lower() == token.lower():
                if column not in attributes:
                    attributes[column] = 0
                if i > 0 and tokens[i - 1].lower() == "not":
                    attributes[column] -= 2
                elif i > 1 and tokens[i - 2].lower() == "not" and pos_tags[tokens[i - 1]] == "RB": # adverb
                    attributes[column] -= 1
                elif i > 1 and tokens[i - 2].lower() != "not" and pos_tags[tokens[i - 1]] == "RB":
                    attributes[column] += 3
                else:
                    attributes[column] += 2
            if token.lower() in column.lower() and pos_tags[token].startswith("NN"):
                if column not in attributes:
                    attributes[column] = 0
                attributes[column] += 2
            if column.lower() in antonyms:
                if column not in attributes:
                    attributes[column] = 0
                attributes[column] -= 2
    # print(pos_tags["not"])
    return attributes

def get_max_values(file_name):
    df = pd.read_excel(file_name, engine='openpyxl')
    columns = df.columns.drop(["Breed"])
    return {column : df[column].max() for column in columns}

def normalize_row(input_row, max_values):
    return {column : input_row[column] / max_values[column] for column in max_values.keys()}

# -------------------------------------------------

with open("../results/networks/network1000_0.001.json", "r", encoding="utf-8") as f:
    network_json = json.load(f)

network = []
n_inputs = network_json["input_layer_size"]
n_hidden = network_json["hidden_layer_size"]
n_outputs = network_json["output_layer_size"]

hidden_layer = [*network_json["weights"]["input_to_hidden"]]
network.append(hidden_layer)
print(hidden_layer)
output_layer = [*network_json["weights"]["hidden_to_output"]]
network.append(output_layer)

row, breed_encoder = init_input("../xlsx/new_main.xlsx")
data_max_values = get_max_values("../xlsx/new_main.xlsx")
print(row, end="\n\n")


# User interaction:
while True:
    input_row = copy.copy(row)
    cat_desc = input("\n> ")
    if cat_desc:
        detected_language = identify_lang(cat_desc)

        tokens = word_tokenize(cat_desc)
        pos_tags = {k: v for k, v in nltk.pos_tag(tokens)}
        df = pd.read_excel("../xlsx/new_main.xlsx", engine='openpyxl')
        columns = df.columns.drop(["Breed"])

        desc_attr = analyze_description(columns, tokens, pos_tags)
        print("\t", desc_attr)

        for column, value_to_add in input_row.items():
            if column in desc_attr.keys():
                input_row[column] = max(min(desc_attr[column], 5), 0)
            else:
                 input_row[column] = round(input_row[column])

        input_row = normalize_row(input_row, data_max_values)

        print("\t", input_row)

        predicted_breed_id = backpropagation.predict(network, list(input_row.values()))
        breed_name = breed_encoder.inverse_transform([predicted_breed_id - 1])[0]

        print(f"> Predicted breed: {breed_name}")
        os.startfile(os.path.dirname(os.getcwd()) + "/breed_images/" + breed_name + ".png")
        if breed_name != "Not Specified":
            pprint(generate_breed_description(breed_name))
            comparison_breed = input("> Breed to compare with: ")
            pprint(generate_breed_comparison(breed_name, comparison_breed))
        else:
            pprint("The term \"Not Specified\" for a cat breed indicates an unknown or mixed breed "
                "origin, resulting in a wide array of potential appearances and temperaments.")


# ------------------------------------------------
cat_desc = read_file_or_keyboard("cat_desc1.txt")
detected_language = identify_lang(cat_desc)
pprint(cat_desc)
print()

tokens = word_tokenize(cat_desc)
pos_tags = {k: v for k, v in nltk.pos_tag(tokens)}
df = pd.read_excel("../xlsx/new_main.xlsx", engine='openpyxl')
columns = df.columns.drop(["Breed"])

keywords = extract_keywords(cat_desc, detected_language)
pprint(keywords)
print()

desc_attr = analyze_description(columns, tokens, pos_tags)
print(desc_attr)

input_row = copy.copy(row)

for column, value_to_add in input_row.items():
    if column in desc_attr.keys():
        input_row[column] = max(min(desc_attr[column], 4), 0)
    else:
         input_row[column] = round(input_row[column])

input_row = normalize_row(input_row, data_max_values)

print(input_row)

predicted_breed_id = backpropagation.predict(network, list(input_row.values()))
breed_name = breed_encoder.inverse_transform([predicted_breed_id - 1])

print(f"> Predicted breed: {breed_name}")

os.startfile(os.path.dirname(os.getcwd()) + "/breed_images/" + breed_name[0] + ".png")

# print("--------------------------------------------")
# identify_lang(text)
# get_stylometric_info(text)
# print("--------------------------------------------")
# detected_language = identify_lang(text)
# replace_words_in_file("cat_desc1.txt", "processed_cat_desc.txt", replacement_fraction=0.2)
# print("--------------------------------------------")
# extract_keywords_and_generate_sentences(text, detected_language)

