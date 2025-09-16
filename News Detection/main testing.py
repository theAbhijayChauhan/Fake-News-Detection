from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import feedparser
import re
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Any
import logging
from datetime import datetime
import urllib.parse
import json
from pathlib import Path
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Fact Checker", version="1.0.0")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the sentence transformer model
try:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    logger.info("Sentence transformer model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load sentence transformer model: {e}")
    model = None

# Data storage files
USERS_FILE = "users.txt"
SEARCH_FILE = "search.txt"

class ClaimRequest(BaseModel):
    claim: str
    user_email: str = None

class User(BaseModel):
    id: str
    name: str
    email: str
    created: str

class SearchHistoryItem(BaseModel):
    id: str
    user_email: str
    claim: str
    verdict: str
    confidence: float
    timestamp: str

class NewsItem(BaseModel):
    title: str
    description: str
    link: str
    source: str
    similarity: float

class FactCheckResponse(BaseModel):
    verdict: str
    confidence: float
    analysis_summary: str
    news_sources: List[NewsItem]

def ensure_files_exist():
    """Create the data files if they don't exist"""
    if not Path(USERS_FILE).exists():
        with open(USERS_FILE, 'w') as f:
            f.write("")
    
    if not Path(SEARCH_FILE).exists():
        with open(SEARCH_FILE, 'w') as f:
            f.write("")

def read_users():
    """Read all users from the users file"""
    ensure_files_exist()
    users = []
    try:
        with open(USERS_FILE, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    # Parse the line format: ID: x | Name: y | Email: z | Created: w
                    parts = line.split(' | ')
                    if len(parts) == 4:
                        user_data = {}
                        for part in parts:
                            key, value = part.split(': ', 1)
                            user_data[key.lower()] = value
                        
                        users.append(User(
                            id=user_data.get('id', ''),
                            name=user_data.get('name', ''),
                            email=user_data.get('email', ''),
                            created=user_data.get('created', '')
                        ))
    except Exception as e:
        logger.error(f"Error reading users: {e}")
    return users

def save_user(user: User):
    """Save a user to the users file"""
    ensure_files_exist()
    try:
        with open(USERS_FILE, 'a') as f:
            f.write(f"ID: {user.id} | Name: {user.name} | Email: {user.email} | Created: {user.created}\n")
        return True
    except Exception as e:
        logger.error(f"Error saving user: {e}")
        return False

def find_user_by_email(email: str):
    """Find a user by email"""
    users = read_users()
    for user in users:
        if user.email == email:
            return user
    return None

def save_search_history(item: SearchHistoryItem):
    """Save a search history item to the search file"""
    ensure_files_exist()
    try:
        with open(SEARCH_FILE, 'a') as f:
            f.write(f"ID: {item.id} | User: {item.user_email} | Claim: {item.claim} | Verdict: {item.verdict} | Confidence: {item.confidence} | Timestamp: {item.timestamp}\n")
        return True
    except Exception as e:
        logger.error(f"Error saving search history: {e}")
        return False

def get_search_history(user_email: str):
    """Get search history for a user"""
    ensure_files_exist()
    history = []
    try:
        with open(SEARCH_FILE, 'r') as f:
            for line in f:
                line = line.strip()
                if line and f"User: {user_email}" in line:
                    # Parse the line format
                    parts = line.split(' | ')
                    if len(parts) >= 6:
                        history_data = {}
                        for part in parts:
                            key, value = part.split(': ', 1)
                            history_data[key.lower()] = value
                        
                        history.append(SearchHistoryItem(
                            id=history_data.get('id', ''),
                            user_email=history_data.get('user', ''),
                            claim=history_data.get('claim', ''),
                            verdict=history_data.get('verdict', ''),
                            confidence=float(history_data.get('confidence', 0)),
                            timestamp=history_data.get('timestamp', '')
                        ))
    except Exception as e:
        logger.error(f"Error reading search history: {e}")
    return history

def extract_keywords(claim: str) -> str:
    """Extract relevant keywords from the claim for search"""
    # Remove common stopwords and keep important words
    stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'was', 'are', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
    
    # Clean the claim and extract words
    words = re.findall(r'\b[a-zA-Z]+\b', claim.lower())
    keywords = [word for word in words if word not in stopwords and len(word) > 2]
    
    # Take the most relevant keywords (limit to avoid overly long queries)
    return ' '.join(keywords[:6])

def fetch_google_news(query: str, max_results: int = 15) -> List[Dict[str, Any]]:
    """Fetch news from Google News RSS feed"""
    try:
        # Encode the query for URL
        encoded_query = urllib.parse.quote_plus(query)
        rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
        
        logger.info(f"Fetching news from: {rss_url}")
        
        # Parse RSS feed
        feed = feedparser.parse(rss_url)
        
        if feed.bozo:
            logger.warning("RSS feed parsing encountered issues")
        
        news_items = []
        for entry in feed.entries[:max_results]:
            # Extract source from the title (Google News format: "Title - Source")
            title_parts = entry.title.split(' - ')
            if len(title_parts) > 1:
                title = ' - '.join(title_parts[:-1])
                source = title_parts[-1]
            else:
                title = entry.title
                source = "Unknown Source"
            
            news_item = {
                'title': title,
                'description': getattr(entry, 'summary', ''),
                'link': getattr(entry, 'link', ''),
                'source': source,
                'published': getattr(entry, 'published', '')
            }
            news_items.append(news_item)
        
        logger.info(f"Found {len(news_items)} news items")
        return news_items
    
    except Exception as e:
        logger.error(f"Error fetching Google News: {e}")
        return []

def calculate_semantic_similarity(claim: str, news_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Calculate semantic similarity between claim and news items"""
    if not model:
        logger.error("Sentence transformer model not available")
        # Return news items with default similarity scores
        for item in news_items:
            item['similarity'] = 0.0
        return news_items
    
    try:
        # Encode the claim
        claim_embedding = model.encode([claim])
        
        # Prepare news texts for comparison
        news_texts = []
        for item in news_items:
            # Combine title and description for better semantic understanding
            text = f"{item['title']} {item.get('description', '')}"
            news_texts.append(text)
        
        if not news_texts:
            return news_items
        
        # Encode all news texts
        news_embeddings = model.encode(news_texts)
        
        # Calculate cosine similarities
        similarities = np.dot(claim_embedding, news_embeddings.T).flatten()
        
        # Add similarity scores to news items
        for i, item in enumerate(news_items):
            if i < len(similarities):
                item['similarity'] = float(similarities[i])
            else:
                item['similarity'] = 0.0
        
        # Sort by similarity (highest first)
        news_items.sort(key=lambda x: x['similarity'], reverse=True)
        
        return news_items
    
    except Exception as e:
        logger.error(f"Error calculating semantic similarity: {e}")
        # Return news items with default similarity scores
        for item in news_items:
            item['similarity'] = 0.0
        return news_items

def analyze_claim_against_news(claim: str, news_items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze the claim against news items and provide verdict"""
    if not news_items:
        return {
            'verdict': 'Unverified',
            'confidence': 0.0,
            'analysis_summary': 'No recent news found related to this claim. Unable to verify.'
        }
    
    # Get similarity scores
    similarities = [item.get('similarity', 0.0) for item in news_items]
    
    if not similarities or max(similarities) == 0.0:
        return {
            'verdict': 'Unverified',
            'confidence': 0.1,
            'analysis_summary': 'No semantically similar news found. This claim may be unverified or relates to very recent events not yet covered by major news sources.'
        }
    
    # Analyze the top similarities
    high_similarity_items = [item for item in news_items if item.get('similarity', 0.0) > 0.5]
    medium_similarity_items = [item for item in news_items if 0.3 <= item.get('similarity', 0.0) <= 0.5]
    
    max_similarity = max(similarities)
    avg_top_3_similarity = np.mean(similarities[:3]) if len(similarities) >= 3 else np.mean(similarities)
    
    # Determine verdict based on similarity patterns
    if max_similarity > 0.7 and len(high_similarity_items) >= 2:
        verdict = 'Likely True'
        confidence = min(0.9, max_similarity + 0.1)
        summary = f"Found {len(high_similarity_items)} highly similar news reports supporting this claim. The highest similarity score was {max_similarity:.1%}."
    elif max_similarity > 0.6:
        verdict = 'Possibly True'
        confidence = max_similarity * 0.8
        summary = f"Found some news reports with moderate similarity to this claim. The highest similarity score was {max_similarity:.1%}. Verification recommended."
    elif max_similarity > 0.4 and len(medium_similarity_items) >= 2:
        verdict = 'Unverified'
        confidence = max_similarity * 0.6
        summary = f"Found some related news but with lower similarity scores. This claim requires further investigation."
    else:
        verdict = 'Likely False'
        confidence = 0.3 + (0.4 - max_similarity) * 0.5 if max_similarity < 0.4 else 0.3
        summary = f"No highly similar news reports found. The claim may be false or unsubstantiated based on current news sources."
    
    # Look for contradictory patterns
    contradictory_keywords = ['not', 'never', 'false', 'denies', 'refutes', 'contrary', 'opposite']
    for item in high_similarity_items[:3]:
        text = f"{item['title']} {item.get('description', '')}".lower()
        if any(keyword in text for keyword in contradictory_keywords):
            if verdict == 'Likely True':
                verdict = 'Contradictory Reports'
                confidence *= 0.7
                summary += " However, some sources may contradict this claim."
    
    return {
        'verdict': verdict,
        'confidence': confidence,
        'analysis_summary': summary
    }

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "AI Fact Checker API is running", "model_loaded": model is not None}

@app.post("/register")
async def register(user_data: dict):
    """Register a new user"""
    try:
        name = user_data.get('name')
        email = user_data.get('email')
        password = user_data.get('password')  # In a real app, you'd hash this
        
        if not name or not email or not password:
            raise HTTPException(status_code=400, detail="Name, email, and password are required")
        
        # Check if user already exists
        existing_user = find_user_by_email(email)
        if existing_user:
            raise HTTPException(status_code=400, detail="User with this email already exists")
        
        # Create new user
        user_id = str(uuid.uuid4())
        created_date = datetime.now().strftime("%Y-%m-%d")
        
        new_user = User(
            id=user_id,
            name=name,
            email=email,
            created=created_date
        )
        
        # Save user
        if save_user(new_user):
            return {"message": "User created successfully", "user": new_user}
        else:
            raise HTTPException(status_code=500, detail="Failed to save user")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in register: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/login")
async def login(credentials: dict):
    """Login endpoint - for demo purposes only"""
    try:
        email = credentials.get('email')
        password = credentials.get('password')
        
        if not email or not password:
            raise HTTPException(status_code=400, detail="Email and password are required")
        
        # In a real app, you'd verify the password against a hashed version
        # For demo purposes, we'll just check if the user exists
        user = find_user_by_email(email)
        
        if user:
            return {"message": "Login successful", "user": user}
        else:
            # Auto-create user for demo purposes
            user_id = str(uuid.uuid4())
            created_date = datetime.now().strftime("%Y-%m-%d")
            name = email.split('@')[0]
            
            new_user = User(
                id=user_id,
                name=name,
                email=email,
                created=created_date
            )
            
            if save_user(new_user):
                return {"message": "User created and logged in successfully", "user": new_user}
            else:
                raise HTTPException(status_code=500, detail="Failed to create user")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in login: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/check-fact", response_model=FactCheckResponse)
async def check_fact(request: ClaimRequest):
    """Main endpoint to fact-check a claim"""
    try:
        logger.info(f"Received claim: {request.claim}")
        
        if not request.claim or len(request.claim.strip()) < 3:
            raise HTTPException(status_code=400, detail="Claim must be at least 3 characters long")
        
        # Step 1: Extract keywords from the claim
        keywords = extract_keywords(request.claim)
        logger.info(f"Extracted keywords: {keywords}")
        
        # Step 2: Fetch recent news
        news_items = fetch_google_news(keywords)
        
        if not news_items:
            logger.warning("No news items found")
            return FactCheckResponse(
                verdict="Unverified",
                confidence=0.0,
                analysis_summary="No recent news found related to this claim. This may be a very recent event or the claim may not be newsworthy.",
                news_sources=[]
            )
        
        # Step 3: Calculate semantic similarities
        news_with_similarity = calculate_semantic_similarity(request.claim, news_items)
        
        # Step 4: Analyze and determine verdict
        analysis = analyze_claim_against_news(request.claim, news_with_similarity)
        
        # Step 5: Prepare response
        top_news = news_with_similarity[:10]  # Return top 10 most relevant
        news_sources = [
            NewsItem(
                title=item['title'],
                description=item.get('description', 'No description available'),
                link=item.get('link', ''),
                source=item.get('source', 'Unknown Source'),
                similarity=item.get('similarity', 0.0)
            )
            for item in top_news
        ]
        
        logger.info(f"Analysis complete. Verdict: {analysis['verdict']}, Confidence: {analysis['confidence']:.2f}")
        
        # Save search history if user_email is provided
        if request.user_email:
            search_id = str(uuid.uuid4())
            timestamp = datetime.now().isoformat()
            
            search_item = SearchHistoryItem(
                id=search_id,
                user_email=request.user_email,
                claim=request.claim,
                verdict=analysis['verdict'],
                confidence=analysis['confidence'],
                timestamp=timestamp
            )
            
            save_search_history(search_item)
        
        return FactCheckResponse(
            verdict=analysis['verdict'],
            confidence=analysis['confidence'],
            analysis_summary=analysis['analysis_summary'],
            news_sources=news_sources
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in check_fact: {e}")
        raise HTTPException(status_code=500, detail="Internal server error occurred while processing the claim")

@app.get("/search-history")
async def get_user_search_history(user_email: str):
    """Get search history for a user"""
    try:
        history = get_search_history(user_email)
        return {"history": history}
    except Exception as e:
        logger.error(f"Error getting search history: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    import uvicorn
    print("Starting AI Fact Checker API...")
    print("Make sure you have installed: pip install fastapi uvicorn sentence-transformers feedparser numpy")
    uvicorn.run(app, host="0.0.0.0", port=8000)
# taken help from AI also!
