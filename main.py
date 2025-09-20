from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from datetime import datetime

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# File path for storing user data
USERS_FILE = "users.txt"
HISTORY_FILE = "history.txt"

def read_users():
    """Read users from the text file"""
    users = []
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "r") as file:
            for line in file:
                if line.strip():
                    parts = line.strip().split(" | ")
                    user = {}
                    for part in parts:
                        key, value = part.split(": ", 1)
                        user[key.lower()] = value
                    users.append(user)
    return users

def write_user(user_data):
    """Write a new user to the text file"""
    with open(USERS_FILE, "a") as file:
        file.write(f"ID: {user_data['id']} | Name: {user_data['name']} | Email: {user_data['email']} | Created: {user_data['created']}\n")

def add_to_history(search_data):
    """Add a search to the history file"""
    with open(HISTORY_FILE, "a") as file:
        file.write(f"User: {search_data['email']} | Date: {search_data['date']} | Search: {search_data['search']} | Credibility: {search_data['credibility']}\n")

def get_user_history(email):
    """Get search history for a specific user"""
    history = []
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as file:
            for line in file:
                if line.strip() and f"User: {email}" in line:
                    parts = line.strip().split(" | ")
                    history_item = {}
                    for part in parts:
                        key, value = part.split(": ", 1)
                        history_item[key.lower()] = value
                    history.append(history_item)
    return history

@app.route('/register', methods=['POST'])
def register():
    try:
        data = request.json
        name = data.get('name')
        email = data.get('email')
        password = data.get('password')
        
        # Read existing users
        users = read_users()
        
        # Check if email already exists
        for user in users:
            if user['email'] == email:
                return jsonify({'error': 'Email already registered'}), 400
        
        # Create new user
        new_id = len(users) + 1
        new_user = {
            'id': new_id,
            'name': name,
            'email': email,
            'created': datetime.now().strftime("%Y-%m-%d")
        }
        
        # Save user to file
        write_user(new_user)
        
        return jsonify({'message': 'User registered successfully', 'user': new_user}), 201
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/login', methods=['POST'])
def login():
    try:
        data = request.json
        email = data.get('email')
        password = data.get('password')
        
        # Read users from file
        users = read_users()
        
        # Find user with matching email
        for user in users:
            if user['email'] == email:
                # In a real app, you would verify the password here
                return jsonify({'message': 'Login successful', 'user': user}), 200
        
        return jsonify({'error': 'Invalid credentials'}), 401
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/add_history', methods=['POST'])
def add_history():
    try:
        data = request.json
        email = data.get('email')
        search_text = data.get('search_text')
        credibility = data.get('credibility', 'N/A')
        
        # Add to history
        search_data = {
            'email': email,
            'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'search': search_text,
            'credibility': credibility
        }
        
        add_to_history(search_data)
        
        return jsonify({'message': 'Search added to history'}), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_history', methods=['GET'])
def get_history():
    try:
        email = request.args.get('email')
        
        # Get user history
        history = get_user_history(email)
        
        return jsonify({'history': history}), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/check_news', methods=['POST'])
def check_news():
    try:
        data = request.json
        news_text = data.get('text', '')
        
        # This is where you would implement your news checking logic
        # For now, we'll return a mock response
        
        # Simple mock analysis based on text length
        text_length = len(news_text)
        if text_length < 50:
            credibility = 20
            assessment = "Likely False"
        elif text_length < 100:
            credibility = 50
            assessment = "Mixed Reliability"
        else:
            credibility = 80
            assessment = "Likely Credible"
        
        return jsonify({
            'credibility': credibility,
            'assessment': assessment,
            'sources_checked': [
                "Reuters News API",
                "Associated Press",
                "Fact-checking databases"
            ]
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Create files if they don't exist
    if not os.path.exists(USERS_FILE):
        open(USERS_FILE, 'w').close()
    
    if not os.path.exists(HISTORY_FILE):
        open(HISTORY_FILE, 'w').close()
    
    app.run(debug=True, port=5000)
