from newsapi import NewsApiClient
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import re

# Initialize NewsAPI
newsapi = NewsApiClient(api_key='75200db5e571465b9650cc1050a8470c')  # Replace with your key

# Load BERT  (BERT model)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Function to get BERT embedding
def get_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# Clean text
def clean(text):
    return re.sub(r'\W+', ' ', text.lower())

# Get user input
user_input = input("🗞️ Enter a news claim: ")
query = clean(user_input)

# Fetch related articles
articles = newsapi.get_everything(q=query, language='en', sort_by='relevancy', page_size=5)

# Get embedding for user input
user_vec = get_embedding(user_input)

# Compare with articles
print("\n🔍 Comparing with real news...\n")
similarities = []
for i, article in enumerate(articles['articles'], 1):
    title = article['title']
    description = article['description'] or ""
    combined = title + " " + description
    article_vec = get_embedding(combined)
    score = cosine_similarity([user_vec], [article_vec])[0][0]
    similarities.append((score, title))

# Sort and show top matches
similarities.sort(reverse=True)
for i, (score, title) in enumerate(similarities, 1):
    print(f"{i}. Similarity: {score:.2f} → {title[:80]}")

# Verdict
top_score = similarities[0][0]
print("\n🧠 Verdict:")
if top_score > 0.75:
    print("✅ This claim is likely supported by real news.")
elif top_score > 0.5:
    print("🤔 This claim is somewhat related but not clearly supported.")
else:
    print("⚠️ This claim is likely NOT supported by current news.")
