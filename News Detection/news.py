import re
import nltk
from newsapi import NewsApiClient
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

# Download stopwords
# nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Initialize NewsAPI
newsapi = NewsApiClient(api_key='75200db5e571465b9650cc1050a8470c')  # Replace with your actual key

# Function to clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    tokens = [word for word in text.split() if word not in stop_words]
    return " ".join(tokens)

# Simulated fake news detection using keyword presence
def is_fake(text):
    fake_keywords = ['shocking', 'click here', 'you won’t believe', 'miracle', 'hoax', 'conspiracy']
    return any(keyword in text.lower() for keyword in fake_keywords)

# Ask user for input
user_input = input("🗞️ Enter a news headline or snippet: ")
query = clean_text(user_input)

# Fetch related articles from NewsAPI
articles = newsapi.get_everything(q=query, language='en', sort_by='relevancy', page_size=5)

# Analyze fetched articles
print("\n🔍 Checking related articles...\n")
found_fake = False
for i, article in enumerate(articles['articles'], 1):
    content = article['title'] + " " + (article['description'] or "")
    cleaned = clean_text(content)
    label = 'FAKE' if is_fake(cleaned) else 'REAL'
    print(f"{i}. {article['title'][:60]}... → {label}")
    if label == 'FAKE':
        found_fake = True

# Final verdict based on related articles
print("\n🧠 Verdict:")
if found_fake:
    print("⚠️ This news might be FAKE based on related articles.")
else:
    print("✅ This news appears to be REAL based on related articles.")
