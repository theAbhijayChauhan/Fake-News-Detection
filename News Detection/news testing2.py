import re
from newsapi import NewsApiClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 🔑 Replace with your actual NewsAPI key
newsapi = NewsApiClient(api_key='75200db5e571465b9650cc1050a8470c')

# 🧹 Clean text
def clean(text):
    return re.sub(r'\W+', ' ', text.lower())

# 🧠 Support and Refute Phrases
support_phrases = ['confirmed', 'official', 'report says', 'announced', 'statement', 'verified']
refute_phrases = ['rumor', 'false', 'debunked', 'not true', 'hoax', 'misleading']

# 📊 Bayesian-style scoring with error handling
def bayesian_score(support, refute):
    total = support + refute
    if total == 0:
        return 0.5  # Neutral confidence when no evidence is found

    p_true = 0.5
    p_evidence_true = support / total
    p_evidence_false = refute / total

    denominator = (p_evidence_true * p_true) + (p_evidence_false * (1 - p_true))
    if denominator == 0:
        return 0.5  # Avoid division by zero, return neutral

    posterior = (p_evidence_true * p_true) / denominator
    return posterior

# 🗞️ Get user input
user_claim = input("🗞️ Enter a news claim: ")
query = clean(user_claim)

# 🔍 Fetch articles
articles = newsapi.get_everything(q=query, language='en', sort_by='relevancy', page_size=5)

# 🧠 TF-IDF + Cosine Similarity
texts = []
titles = []
for article in articles['articles']:
    title = article['title']
    description = article['description'] or ""
    full_text = clean(title + " " + description)
    texts.append(full_text)
    titles.append(title)

vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform([user_claim] + texts)
similarities = cosine_similarity(vectors[0:1], vectors[1:]).flatten()

# 🔎 Phrase scoring
support_score = 0
refute_score = 0
for text in texts:
    support_score += sum(text.count(p) for p in support_phrases)
    refute_score += sum(text.count(p) for p in refute_phrases)

# 🧠 Bayesian verdict
confidence = bayesian_score(support_score, refute_score)

# 📋 Show results
print("\n🔍 Article Similarities:")
for i, (title, sim) in enumerate(zip(titles, similarities), 1):
    print(f"{i}. Similarity: {sim:.2f} → {title[:80]}")

print("\n📊 Phrase Scores:")
print(f"Support phrases found: {support_score}")
print(f"Refute phrases found: {refute_score}")

print("\n🧠 Verdict:")
if confidence > 0.75:
    print("✅ This claim is likely TRUE based on current news.")
elif confidence < 0.3:
    print("⚠️ This claim is likely FALSE or misleading.")
else:
    print("🤔 This claim is unclear or partially supported.")
