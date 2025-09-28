from flask import Flask, request, render_template_string, jsonify
from flask_cors import CORS
import requests
import nltk
import sqlite3
import hashlib
import json
from datetime import datetime, timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from config import NEWS_API_KEY
import re
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize database
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            subscription TEXT DEFAULT 'Free',
            usage_count INTEGER DEFAULT 5,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_reset DATE DEFAULT CURRENT_DATE
        )
    ''')
    conn.commit()
    conn.close()

def hash_password(password):
    """Simple password hashing"""
    return hashlib.sha256(password.encode()).hexdigest()

def get_user(email):
    """Get user from database"""
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('SELECT * FROM users WHERE email = ?', (email,))
    user = c.fetchone()
    conn.close()
    
    if user:
        return {
            'id': user[0],
            'email': user[1],
            'password_hash': user[2],
            'subscription': user[3],
            'usage_count': user[4],
            'created_at': user[5],
            'last_reset': user[6]
        }
    return None

def create_user(email, password):
    """Create new user"""
    try:
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        password_hash = hash_password(password)
        c.execute('INSERT INTO users (email, password_hash) VALUES (?, ?)', 
                 (email, password_hash))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        return False

def update_user_subscription(email, subscription='Premium'):
    """Update user subscription"""
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('UPDATE users SET subscription = ? WHERE email = ?', 
             (subscription, email))
    conn.commit()
    conn.close()

def update_user_usage(email, usage_count):
    """Update user usage count"""
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('UPDATE users SET usage_count = ? WHERE email = ?', 
             (usage_count, email))
    conn.commit()
    conn.close()

def reset_daily_usage():
    """Reset usage count for all free users (call this daily)"""
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    today = datetime.now().date()
    c.execute('''UPDATE users 
                 SET usage_count = 5, last_reset = ? 
                 WHERE subscription = 'Free' AND last_reset < ?''', 
             (today, today))
    conn.commit()
    conn.close()

class NewsVerifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english')
        
    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text
    
    def query_keywords(self, statement):
        try:
            stop_words = set(nltk.corpus.stopwords.words('english'))
        except LookupError:
            # Download stopwords if not available
            nltk.download('stopwords')
            stop_words = set(nltk.corpus.stopwords.words('english'))
            
        question_words = {'is', 'are', 'was', 'were', 'did', 'do', 'does', 'has', 'have', 'had','can', 'could', 'should', 'would', 'what', 'when', 'where', 'who', 'why', 'how'}
        
        try:
            words = nltk.word_tokenize(statement.lower())
        except LookupError:
            # Download punkt if not available
            nltk.download('punkt')
            words = nltk.word_tokenize(statement.lower())
            
        filtered = [w for w in words if w.isalpha() and w not in stop_words and w not in question_words]
        return ' '.join(filtered)

    def get_news_articles(self, query):
        url = f"https://newsapi.org/v2/everything?q={query}&apiKey={NEWS_API_KEY}&pageSize=100"
        try:
            response = requests.get(url)
            if response.status_code == 200:
                articles = [
                    (article['title'] or '') + " " + (article['description'] or '')
                            for article in response.json().get('articles', [])]
                print(f"DEBUG: Retrieved {len(articles)} articles for query '{query}'")
                return articles
            else:
                print("Error fetching news:", response.json())
                return []
        except Exception as e:
            print(f"Error fetching news: {e}")
            return []
    
    def verify_statement(self, statement):
        # Get news articles related to the statement
        query = self.query_keywords(statement)
        articles = self.get_news_articles(query)
        
        print(f"DEBUG: Sample articles fetched:")
        for art in articles[:3]:
            print(art[:100])
            
        if not articles:
            return False, 0.0, 0, []
            
        # Preprocess texts
        clean_statement = self.clean_text(statement)
        clean_articles = [self.clean_text(article) for article in articles]
        
        try:
            # Vectorize and compare
            tfidf_matrix = self.vectorizer.fit_transform([clean_statement] + clean_articles)
            similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
            
            max_similarity = similarities.max()
            threshold = 0.05
            
            return max_similarity > threshold, float(max_similarity * 100), len(articles), articles[:5]
        except Exception as e:
            print(f"Error in verification: {e}")
            return False, 0.0, len(articles), articles[:5]

# Initialize
init_db()
verifier = NewsVerifier()

@app.route('/')
def home():
    return """
    <h1>Backend is running!</h1>
    <p>Use the frontend HTML file to access the system.</p>
    <p>Database initialized successfully!</p>
    """

@app.route('/register', methods=['POST'])
def register():
    try:
        data = request.get_json()
        email = data.get('email', '').strip().lower()
        password = data.get('password', '')
        
        if not email or not password:
            return jsonify({'success': False, 'message': 'Email and password required'})
        
        if len(password) < 4:
            return jsonify({'success': False, 'message': 'Password must be at least 4 characters'})
            
        if create_user(email, password):
            return jsonify({'success': True, 'message': 'User registered successfully'})
        else:
            return jsonify({'success': False, 'message': 'Email already exists'})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Registration error: {str(e)}'})

@app.route('/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        email = data.get('email', '').strip().lower()
        password = data.get('password', '')
        
        if not email or not password:
            return jsonify({'success': False, 'message': 'Email and password required'})
        
        user = get_user(email)
        
        if user and user['password_hash'] == hash_password(password):
            # Check if we need to reset daily usage
            today = datetime.now().date()
            if user['subscription'] == 'Free' and str(today) != user['last_reset']:
                # Reset usage count for new day
                update_user_usage(email, 5)
                conn = sqlite3.connect('users.db')
                c = conn.cursor()
                c.execute('UPDATE users SET last_reset = ? WHERE email = ?', (today, email))
                conn.commit()
                conn.close()
                user['usage_count'] = 5
            
            return jsonify({
                'success': True, 
                'user': {
                    'email': user['email'],
                    'subscription': user['subscription'],
                    'usage_count': user['usage_count']
                }
            })
        else:
            return jsonify({'success': False, 'message': 'Invalid email or password'})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Login error: {str(e)}'})

@app.route('/subscribe', methods=['POST'])
def subscribe():
    try:
        data = request.get_json()
        email = data.get('email', '').strip().lower()
        
        user = get_user(email)
        if not user:
            return jsonify({'success': False, 'message': 'User not found'})
        
        if user['subscription'] == 'Premium':
            return jsonify({'success': False, 'message': 'Already premium user'})
        
        # In a real app, you'd handle payment here
        update_user_subscription(email, 'Premium')
        
        return jsonify({'success': True, 'message': 'Subscribed to Premium successfully!'})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Subscription error: {str(e)}'})

@app.route('/update_usage', methods=['POST'])
def update_usage():
    try:
        data = request.get_json()
        email = data.get('email', '').strip().lower()
        usage_count = data.get('usage_count', 0)
        
        update_user_usage(email, usage_count)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Usage update error: {str(e)}'})

@app.route('/verify', methods=['POST'])
def verify():
    try:
        claim = request.form.get('claim', '')
        user_email = request.form.get('user_email', '')
        
        if not claim:
            return render_result_page("", False, 0.0, 0, [], "No claim provided")
        
        # Check user's usage (additional backend validation)
        if user_email:
            user = get_user(user_email)
            if user and user['subscription'] == 'Free' and user['usage_count'] <= 0:
                return render_result_page(claim, False, 0.0, 0, [], 
                                        "Daily limit reached! Subscribe to Premium for unlimited checks.")
        
        # Perform verification
        is_true, confidence, article_count, sample_articles = verifier.verify_statement(claim)
        
        return render_result_page(claim, is_true, confidence, article_count, sample_articles)
    
    except Exception as e:
        return render_result_page("", False, 0.0, 0, [], f"Verification error: {str(e)}")

def render_result_page(claim, is_true, confidence, article_count, sample_articles, error_msg=None):
    # Create HTML for sample articles
    articles_html = ""
    
    if error_msg:
        articles_html = f"""
        <div class="bg-red-50 border border-red-200 p-4 rounded-lg">
            <p class="text-red-800">
                <i class="fas fa-exclamation-triangle mr-2"></i>
                {error_msg}
            </p>
        </div>
        """
    elif article_count > 0:
        articles_html = f"""
        <div class="mt-6 pt-4 border-t border-gray-200">
            <h3 class="font-bold text-gray-800 mb-3 flex items-center">
                <i class="fas fa-newspaper mr-2 text-indigo-500"></i>
                Searched {article_count} Articles
            </h3>
            <div class="bg-gray-50 p-4 rounded-lg">
                <h4 class="font-medium text-gray-700 mb-2">Sample Articles:</h4>
                <div class="space-y-2 text-sm">
        """
        
        for i, article in enumerate(sample_articles[:3]):
            articles_html += f"""
                    <div class="flex items-start">
                        <span class="bg-indigo-100 text-indigo-600 rounded-full w-5 h-5 flex items-center justify-center text-xs mr-2 mt-1">{i+1}</span>
                        <span class="text-gray-600">"{article[:120]}..."</span>
                    </div>
            """
        
        articles_html += """
                </div>
            </div>
        </div>
        """
    else:
        articles_html = '<p class="text-red-500">No articles found for verification</p>'
    
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Verification Result - NewsCheck</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    </head>
    <body class="bg-gradient-to-br from-gray-100 to-gray-200 p-8 min-h-screen">
        <div class="max-w-4xl mx-auto">
            <!-- Header -->
            <div class="bg-white rounded-t-lg shadow-lg p-6 border-b-4 border-indigo-500">
                <div class="flex items-center justify-between">
                    <div>
                        <h1 class="text-3xl font-bold text-indigo-600 mb-2">NewsCheck Results</h1>
                        <p class="text-gray-600">AI-Powered News Verification Analysis</p>
                    </div>
                    <i class="fas fa-shield-alt text-4xl text-indigo-500"></i>
                </div>
            </div>
            
            <!-- Main Results -->
            <div class="bg-white shadow-lg p-6 space-y-6">
                <div class="bg-gray-50 p-4 rounded-lg border-l-4 border-indigo-400">
                    <h2 class="font-bold text-gray-800 mb-2">Statement Analyzed:</h2>
                    <p class="text-gray-700 italic">"{claim}"</p>
                </div>
                
                {"" if error_msg else f'''
                <div class="grid md:grid-cols-3 gap-6">
                    <div class="text-center p-6 rounded-lg {'bg-green-50 border-2 border-green-200' if is_true else 'bg-red-50 border-2 border-red-200'}">
                        <div class="text-4xl mb-3">
                            {'✅' if is_true else '❌'}
                        </div>
                        <h3 class="font-bold text-lg {'text-green-700' if is_true else 'text-red-700'}">
                            {'LIKELY TRUE' if is_true else 'LIKELY FALSE'}
                        </h3>
                        <p class="text-sm text-gray-600 mt-2">Based on news analysis</p>
                    </div>
                    
                    <div class="text-center p-6 rounded-lg bg-blue-50 border-2 border-blue-200">
                        <div class="text-3xl font-bold text-blue-700 mb-2">{confidence:.1f}%</div>
                        <h3 class="font-bold text-lg text-blue-700">Confidence Score</h3>
                        <p class="text-sm text-gray-600 mt-2">Similarity to news sources</p>
                    </div>
                    
                    <div class="text-center p-6 rounded-lg bg-purple-50 border-2 border-purple-200">
                        <div class="text-3xl font-bold text-purple-700 mb-2">{article_count}</div>
                        <h3 class="font-bold text-lg text-purple-700">Articles Searched</h3>
                        <p class="text-sm text-gray-600 mt-2">From various news sources</p>
                    </div>
                </div>
                
                <!-- Confidence Level Indicator -->
                <div class="bg-gray-50 p-4 rounded-lg">
                    <h3 class="font-bold text-gray-800 mb-3">Reliability Assessment:</h3>
                    <div class="flex items-center space-x-2">
                        <span class="text-sm font-medium text-gray-600">Low</span>
                        <div class="flex-1 bg-gray-200 rounded-full h-3 relative">
                            <div class="bg-gradient-to-r from-red-400 via-yellow-400 to-green-400 h-3 rounded-full" 
                                 style="width: 100%;"></div>
                            <div class="absolute top-0 bg-indigo-600 h-3 w-2 rounded-full transform -translate-x-1" 
                                 style="left: {confidence}%;"></div>
                        </div>
                        <span class="text-sm font-medium text-gray-600">High</span>
                    </div>
                    <p class="text-sm text-gray-600 mt-2">
                        {'High reliability - Strong correlation with news sources' if confidence > 70 
                         else 'Medium reliability - Moderate correlation found' if confidence > 40
                         else 'Low reliability - Limited correlation with current news'}
                    </p>
                </div>
                '''}
                
                {articles_html}
                
                <!-- Tips Section -->
                <div class="bg-blue-50 border border-blue-200 p-4 rounded-lg">
                    <h3 class="font-bold text-blue-800 mb-2 flex items-center">
                        <i class="fas fa-lightbulb mr-2"></i>
                        Tips for News Verification:
                    </h3>
                    <ul class="text-sm text-blue-700 space-y-1">
                        <li>• Cross-check with multiple reliable news sources</li>
                        <li>• Look for official statements and primary sources</li>
                        <li>• Check the publication date and author credentials</li>
                        <li>• Be cautious of sensational headlines and emotional language</li>
                    </ul>
                </div>
            </div>
            
            <!-- Footer -->
            <div class="bg-gray-800 rounded-b-lg shadow-lg p-6 text-center">
                <div class="flex justify-center items-center space-x-6 mb-4">
                    <a href="javascript:history.back()" 
                       class="bg-indigo-600 text-white px-6 py-2 rounded-lg hover:bg-indigo-700 transition-all duration-300 flex items-center">
                        <i class="fas fa-arrow-left mr-2"></i>
                        Check Another Claim
                    </a>
                    <button onclick="window.print()" 
                            class="bg-gray-600 text-white px-6 py-2 rounded-lg hover:bg-gray-700 transition-all duration-300 flex items-center">
                        <i class="fas fa-print mr-2"></i>
                        Print Results
                    </button>
                </div>
                
                <div class="text-gray-400 text-sm">
                    <p>© 2024 NewsCheck - Powered by AI and NewsAPI</p>
                    <p class="mt-1">Developed by: Abhijay, Rakesh, Jaya, Abhishek</p>
                </div>
            </div>
        </div>
    </body>
    </html>
    """

if __name__ == "__main__":
    print("Starting NewsCheck Backend...")
    print("Initializing database...")
    
    # Test database connection
    try:
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute('SELECT COUNT(*) FROM users')
        user_count = c.fetchone()[0]
        conn.close()
        print(f"Database ready! Current users: {user_count}")
    except Exception as e:
        print(f"Database error: {e}")
    
    print("Server starting on http://localhost:8000")
    app.run(port=8000, debug=True)
