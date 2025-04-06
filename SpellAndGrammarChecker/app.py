from flask import Flask, render_template, request, jsonify
from googletrans import Translator
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import wordnet
from spellchecker import SpellChecker
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from textblob import TextBlob
import textstat
from wordfreq import word_frequency
import nltk
import random
from collections import Counter
from gtts import gTTS
from pydub import AudioSegment
import os
import uuid



# Download necessary NLTK data
nltk.download('vader_lexicon')
nltk.download('punkt')

app = Flask(__name__)
spell = SpellChecker()
translator = Translator()
sia = SentimentIntensityAnalyzer()

# ✅ Spell Checker
def check_spelling(text):
    words = text.split()
    corrected_words = [spell.correction(word) if spell.correction(word) else word for word in words]
    return " ".join(corrected_words)

# ✅ Auto Capitalization
def auto_capitalize(text):
    sentences = text.split('. ')
    return '. '.join(sentence.capitalize() for sentence in sentences)

# ✅ Synonym Suggestion
def suggest_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return list(synonyms) if synonyms else ["No synonyms found"]

# ✅ Word Counter
def count_words(text):
    words = text.split()
    return f"Word Count: {len(words)}"

# ✅ Syllable Counter
def count_syllables(text):
    return f"Syllable Count: {textstat.syllable_count(text)}"

# ✅ Word Complexity Analysis
# ✅ Word Complexity Analysis (Fixed Formatting)
def analyze_word_complexity(text):
    words = text.split()
    complexity_scores = {word: round(word_frequency(word, 'en', minimum=0.0000001), 8) for word in words}

    if not complexity_scores:
        return "⚠️ No valid words found."

    # Formatting output as a string
    formatted_output = "<b>Word Complexity Scores:</b><br>"
    for word, score in complexity_scores.items():
        formatted_output += f"{word}: {score}<br>"

    return formatted_output


# ✅ Text Summarization
def summarize_text(text, num_sentences=2):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, num_sentences)
    return " ".join(str(sentence) for sentence in summary)

# ✅ Sentiment Analysis
def analyze_sentiment(text):
    sentiment = TextBlob(text).sentiment.polarity
    if sentiment > 0:
        return "😊 Positive"
    elif sentiment < 0:
        return "😡 Negative"
    else:
        return "😐 Neutral"

# ✅ Emotion-Based Tone Analysis
def analyze_tone(text):
    scores = sia.polarity_scores(text)
    compound = scores['compound']

    if compound >= 0.5:
        return "😊 Happy"
    elif 0.2 <= compound < 0.5:
        return "🙂 Content"
    elif -0.2 < compound < 0.2:
        return "😐 Neutral"
    elif -0.5 <= compound <= -0.2:
        return "😞 Sad"
    else:
        return "😡 Angry"

# ✅ Multi-Language Translation
def translate_text(text, target_lang):
    valid_languages = ["en", "es", "fr", "de", "hi", "zh-cn", "ar", "ru"]

    if target_lang not in valid_languages:
        return "⚠️ Translation Error: Invalid destination language."

    try:
        translation = translator.translate(text, dest=target_lang)
        return translation.text if translation.text else "⚠️ Translation Failed."
    except Exception as e:
        return f"⚠️ Translation Error: {str(e)}"

# ✅ Title Generator
def generate_title(text):
    blob = TextBlob(text)
    words = blob.noun_phrases
    word_counts = Counter(blob.words)

    main_topic = words[0].capitalize() if words else (blob.words[0].capitalize() if blob.words else "Untitled")
    common_word = word_counts.most_common(1)[0][0] if word_counts else main_topic

    templates = [
        f"The Power of {main_topic}",
        f"How {main_topic} is Changing the World",
        f"Top 5 Secrets About {main_topic}",
        f"The Ultimate Guide to {main_topic}",
        f"Why {common_word} Matters More Than Ever",
        f"10 Things You Didn’t Know About {main_topic}",
        f"The Future of {main_topic}: What You Need to Know",
        f"{main_topic} Explained: A Beginner’s Guide",
        f"How to Master {main_topic} in 30 Days",
        f"Breaking Down {main_topic}: Insights & Trends"
    ]

    return templates

# ✅ Blog Outline Generator (Unique)
def generate_blog_outline(text):
    blob = TextBlob(text)
    keywords = blob.noun_phrases
    main_topic = keywords[0].capitalize() if keywords else "Your Topic"

    key_points = [
        f"Understanding {main_topic}",
        f"Importance of {main_topic} in Today's World",
        f"Common Challenges in {main_topic}",
        f"Best Practices for {main_topic}",
        f"Latest Innovations in {main_topic}",
        f"How {main_topic} Impacts Society",
        f"Top Myths About {main_topic}",
        f"Case Studies on {main_topic}",
        f"Future Trends of {main_topic}",
        f"How to Get Started with {main_topic}"
    ]

    random.shuffle(key_points)
    selected_points = key_points[:3]

    outline = {
        "Introduction": f"An overview of {main_topic} and why it matters.",
        "Key Points": selected_points,
        "Conclusion": f"Final thoughts on {main_topic} and its future potential."
    }
    return outline

# ✅ Hashtag Generator (Updated for Multiple Hashtags)
def generate_hashtags(text):
    blob = TextBlob(text)
    keywords = blob.noun_phrases  # Extract keywords
    hashtags = [f"#{word.replace(' ', '')}" for word in keywords]  # Convert to hashtags
    random.shuffle(hashtags)  # Shuffle for variety
    return hashtags[:10] if hashtags else ["#NoHashtagsFound"]  # Limit to 10 hashtag
def generate_audio_response(text_main):
    # Create unique ID for this session
    uid = str(uuid.uuid4())

    # Define texts
    intro_text = "Here’s your result."
    outro_text = "Thanks for using WordWise!"

    # Convert text to audio
    def tts_to_mp3(text, filename):
        tts = gTTS(text=text, lang='en')
        tts.save(filename)

    # Filenames
    base_path = "static/audio/"
    os.makedirs(base_path, exist_ok=True)
    intro_file = os.path.join(base_path, f"{uid}_intro.mp3")
    main_file = os.path.join(base_path, f"{uid}_main.mp3")
    outro_file = os.path.join(base_path, f"{uid}_outro.mp3")
    final_file = os.path.join(base_path, f"{uid}_final.mp3")

    # Generate audio parts
    tts_to_mp3(intro_text, intro_file)
    tts_to_mp3(text_main, main_file)
    tts_to_mp3(outro_text, outro_file)

    # Merge them
    intro = AudioSegment.from_mp3(intro_file)
    main = AudioSegment.from_mp3(main_file)
    outro = AudioSegment.from_mp3(outro_file)

    final_audio = intro + AudioSegment.silent(duration=500) + main + AudioSegment.silent(duration=500) + outro
    final_audio.export(final_file, format="mp3")

    return final_file



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    text = request.form.get('text', '')
    option = request.form.get('option', '')
    target_lang = request.form.get('target_lang', 'en')

    if option == 'title_generator':
        result = generate_title(text)
    elif option == 'blog_outline':
        result = generate_blog_outline(text)
    elif option == 'translate_text':
        result = translate_text(text, target_lang)
    elif option == 'tone_analysis':
        result = analyze_tone(text)
    elif option == 'sentiment_analysis':
        result = analyze_sentiment(text)
    elif option == 'spell_checker':
        result = check_spelling(text)
    elif option == 'synonym_suggestion':
        result = {word: suggest_synonyms(word) for word in text.split()}
    elif option == 'auto_capitalize':
        result = auto_capitalize(text)
    elif option == 'word_counter':
        result = count_words(text)
    elif option == 'text_summarization':
        result = summarize_text(text)
    elif option == 'syllable_counter':
        result = count_syllables(text)
    elif option == 'word_complexity':
        result = analyze_word_complexity(text)
    elif option == 'hashtag_generator':  # ✅ Multi-Hashtag Generator
        result = generate_hashtags(text)
    else:
        result = "⚠️ Invalid option selected."
    if isinstance(result, str):
        audio_path = generate_audio_response(result)
        audio_url = f"/{audio_path.replace(os.path.sep, '/')}"
    else:
        audio_url = None

    return jsonify({'result': result, 'audio_url': audio_url})

if __name__ == '__main__':
    app.run(debug=True)
