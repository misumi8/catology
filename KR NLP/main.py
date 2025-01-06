from fileinput import filename
from pprint import pprint
from langdetect import detect, detect_langs
from nltk.tokenize import word_tokenize
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
nltk.download('punkt')

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

text = read_file_or_keyboard("ro_test.txt")
pprint(text)
get_stylometric_info(text)
