from http.client import responses
from pprint import pprint
from langdetect import detect, detect_langs
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import re
import nltk
import matplotlib.pyplot as plt
from rake_nltk import Rake
import google.generativeai as genai
nltk.download('punkt')
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

language_map = {
    'ro': 'romanian',
    'en': 'english',
    'fr': 'french',
    'es': 'spanish',
}

def extract_keywords_and_generate_sentences(text, detected_language):
    if detected_language not in language_map:
        print(f"Stop-words for language {detected_language} are not available.")
        return
    
    stop_words = set(stopwords.words(language_map[detected_language]))

    rake = Rake() # max_length=2
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

text = read_file_or_keyboard("ro.txt")
pprint(text)
identify_lang(text)
get_stylometric_info(text)

detected_language = identify_lang(text)
extract_keywords_and_generate_sentences(text, detected_language)
