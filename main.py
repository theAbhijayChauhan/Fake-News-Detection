from flask import Flask, request
app = Flask(__name__)


import requests
import nltk
import re
from config import NEWS_API_KEY
from sentence_transformers import SentenceTransformer, util

# Load lightweight BERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

class NewsVerifier:
    def __init__(self):
        pass

    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text

    def query_keywords(self, statement):
        # Keep only nouns and proper nouns for better queries
        words = nltk.word_tokenize(statement)
        pos_tags = nltk.pos_tag(words)
        filtered = [w for w, t in pos_tags if t in ('NN', 'NNP', 'NNS', 'NNPS')]
        return ' '.join(filtered) if filtered else statement

    def get_news_articles(self, query):
        url = f"https://newsapi.org/v2/everything?q={query}&apiKey={NEWS_API_KEY}&pageSize=100"
        response = requests.get(url)
        if response.status_code == 200:
            articles = [
                (article.get('title','') + " " + article.get('description','') + " " + article.get('content',''))
                for article in response.json().get('articles', [])
            ]
            print(f"DEBUG: Retrieved {len(articles)} articles for query '{query}'")
            return articles
        else:
            print("Error fetching news:", response.json())
            return []

    def verify_statement(self, statement):
        # Step 1: Extract keywords & fetch articles
        query = self.query_keywords(statement)
        articles = self.get_news_articles(query)

        if not articles:
            return False, 0.0, 0, []

        # Step 2: Clean texts
        clean_statement = self.clean_text(statement)
        clean_articles = [self.clean_text(article) for article in articles]

        # Step 3: Encode with BERT
        embeddings = model.encode([clean_statement] + clean_articles, convert_to_tensor=True)

        # Step 4: Compute cosine similarities
        similarities = util.cos_sim(embeddings[0], embeddings[1:]).cpu().numpy().flatten()

        max_similarity = similarities.max()
        best_idx = similarities.argmax()
        best_article = articles[best_idx]

        # Step 5: Decide based on threshold
        threshold = 0.3  # tuned for BERT
        return max_similarity > threshold, float(max_similarity * 100), len(articles), [best_article]

@app.route('/')
def home():
    return """
    <h1>Backend is running!</h1>
    <p>Use the frontend HTML file to access the system.</p>
    """

@app.route('/verify', methods=['POST'])
def verify():
    claim = request.form.get('claim', '')
    is_true, confidence, article_count, sample_articles = verifier.verify_statement(claim)
    return render_result_page(claim, is_true, confidence, article_count, sample_articles)

def render_result_page(claim, is_true, confidence, article_count, sample_articles):
    articles_html = ""
    if article_count > 0:
        articles_html = f"""
        <div class="mt-6 pt-4 border-t border-gray-200">
            <h3 class="font-bold text-gray-800 mb-3 flex items-center">
                <i class="fas fa-newspaper mr-2 text-indigo-500"></i>
                Searched {article_count} Articles
            </h3>
            <div class="bg-gray-50 p-4 rounded-lg">
                <h4 class="font-medium text-gray-700 mb-2">Most Relevant Article:</h4>
                <div class="space-y-2 text-sm">
        """
        for i, article in enumerate(sample_articles):
            articles_html += f"""
                    <div class="flex items-start">
                        <span class="bg-indigo-100 text-indigo-600 rounded-full w-5 h-5 flex items-center justify-center text-xs mr-2 mt-1">{i+1}</span>
                        <span class="text-gray-600">"{article[:200]}..."</span>
                    </div>
            """
        articles_html += """
                </div>
            </div>
        </div>
        """

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Result</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    </head>
    <body class="bg-gray-100 p-8">
        <div class="max-w-2xl mx-auto bg-white rounded-lg shadow p-6">
            <h1 class="text-2xl font-bold text-indigo-600 mb-4">Verification Result</h1>
            <div class="space-y-4">
                <div>
                    <p class="font-medium">Statement:</p>
                    <p class="text-gray-700">"{claim}"</p>
                </div>
                
                <div class="grid grid-cols-2 gap-4">
                    <div>
                        <p class="font-medium">Verification:</p>
                        <p class="{'text-green-600' if is_true else 'text-red-600'} font-bold text-lg">
                            {'✅ TRUE' if is_true else '❌ FALSE'}
                        </p>
                    </div>
                    <div>
                        <p class="font-medium">Confidence:</p>
                        <p class="text-gray-700 font-bold">{confidence:.2f}%</p>
                    </div>
                </div>
                
                <div class="bg-blue-50 p-3 rounded-lg">
                    <p class="font-medium text-blue-800 flex items-center">
                        <i class="fas fa-database mr-2"></i>
                        Articles Analyzed: 
                        <span class="ml-2 bg-blue-600 text-white px-2 py-1 rounded-full text-sm">{article_count}</span>
                    </p>
                </div>
                
                {articles_html if article_count > 0 else '<p class="text-red-500">No articles found for verification</p>'}
            </div>
            
            <a href="/" class="mt-6 inline-block text-indigo-600 hover:underline">
                <i class="fas fa-arrow-left mr-2"></i>Check another claim
            </a>
        </div>
    </body>
    </html>
    """

if __name__ == "__main__":
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    verifier = NewsVerifier()
    app.run(port=8000, debug=True)

