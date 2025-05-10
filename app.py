from flask import Flask, request, jsonify, render_template, send_from_directory
import pandas as pd
import numpy as np
import random
import os
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

app = Flask(__name__, static_folder='static')

# Download NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

# Global variables to store model and data
vectorizer = None
model_trained = False
faq_df = None
tfidf_matrix = None
ticket_types = None  # Store ticket types for categorization
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

funny_responses = [
    "I don't know anything yet because you haven't trained me! Click the 'Train Model' button to enlighten me.",
    "Error 404: Knowledge not found. Have you tried turning me on? I mean, training me?",
    "My brain is as empty as a developer's wallet after buying a new MacBook. Please train me!",
    "I'm currently as useful as a chocolate teapot. Train me to be more helpful!",
    "Still loading personality... Please click 'Train Model' to accelerate the process.",
]

@app.route('/')
def index():
    """Render the main chat interface."""
    return render_template('index.html')

@app.route('/static/<path:path>')
def serve_static(path):
    """Serve static files."""
    return send_from_directory('static', path)

def preprocess_text(text):
    """Clean and preprocess text data."""
    if isinstance(text, str):
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Tokenize
        tokens = word_tokenize(text)
        # Remove stopwords and lemmatize
        tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
        return ' '.join(tokens)
    return ""

def format_response(answer, ticket_type=None):
    """Format the response to be more helpful and conversational."""
    # Handle empty responses
    if not isinstance(answer, str) or not answer.strip():
        return "I'm sorry, I don't have specific information on that. Could you provide more details about your issue?"
    
    # Clean up the answer
    answer = answer.strip()
    answer = re.sub(r'\s+', ' ', answer)  # Remove extra whitespace
    
    # Add a prefix based on ticket type if available
    prefix = ""
    if ticket_type:
        if 'technical' in ticket_type.lower():
            prefix = "For this technical issue: "
        elif 'billing' in ticket_type.lower():
            prefix = "Regarding your billing concern: "
        elif 'account' in ticket_type.lower():
            prefix = "For your account issue: "
        elif 'product' in ticket_type.lower():
            prefix = "About your product question: "
        else:
            prefix = "Here's how I can help: "
    else:
        prefix = "Here's what I found: "
    
    # Add a helpful suffix
    suffix = " Is there anything else you'd like to know?"
    
    # Format the full response
    formatted_answer = f"{prefix}{answer}{suffix}"
    
    # Make sure the first letter is capitalized
    formatted_answer = formatted_answer[0].upper() + formatted_answer[1:]
    
    return formatted_answer

@app.route('/api/train', methods=['POST'])
def train_model():
    """Train the NLP model on the customer support dataset."""
    global vectorizer, model_trained, faq_df, tfidf_matrix, ticket_types
    
    try:
        # Load the dataset - we assume it's already in the project directory
        data = pd.read_csv('customer_support_tickets.csv')
        
        # Clean and prepare the data
        data = data.dropna(subset=['Ticket Subject', 'Ticket Description', 'Resolution'])
        
        # Store ticket types for context
        ticket_types = data[['Ticket Type', 'Ticket Subject']].copy()
        
        # Preprocess text data
        data['Processed_Subject'] = data['Ticket Subject'].apply(preprocess_text)
        data['Processed_Description'] = data['Ticket Description'].apply(preprocess_text)
        
        # Combine subject and description for better context
        data['Question'] = data['Processed_Subject'] + " " + data['Processed_Description']
        data['Original_Question'] = data['Ticket Subject'] + " " + data['Ticket Description']
        data['Answer'] = data['Resolution']
        data['Ticket_Type'] = data['Ticket Type']
        
        # Create a simplified dataset for question answering
        faq_df = data[['Question', 'Original_Question', 'Answer', 'Ticket_Type']].copy()
        
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform(faq_df['Question'])
        
        # Save the model for future use
        with open('model.pkl', 'wb') as f:
            pickle.dump((vectorizer, faq_df, tfidf_matrix, ticket_types), f)
        
        model_trained = True
        return jsonify({"status": "success", "message": "Model trained successfully!"})
    
    except Exception as e:
        print(f"Training Error: {e}")
        return jsonify({"status": "error", "message": str(e)})

@app.route('/api/chat', methods=['POST'])
def chat():
    """Process user messages and return bot responses."""
    global model_trained, vectorizer, faq_df, tfidf_matrix
    
    user_message = request.json.get('message', '')
    
    if not model_trained:
        # Return a random funny response if model is not trained
        return jsonify({"response": random.choice(funny_responses)})
    
    try:
        # Preprocess the user query
        processed_query = preprocess_text(user_message)
        
        # Transform user query using the vectorizer
        user_vector = vectorizer.transform([processed_query])
        
        # Calculate similarity with all FAQs
        similarity_scores = cosine_similarity(user_vector, tfidf_matrix).flatten()
        
        # Get top 3 matches
        top_indices = similarity_scores.argsort()[-3:][::-1]
        top_scores = [similarity_scores[i] for i in top_indices]
        
        # If best similarity is too low, return a fallback message
        if top_scores[0] < 0.15:
            return jsonify({
                "response": "I'm not sure I understand your specific issue. Could you please provide more details about what problem you're experiencing with the product or service?"
            })
        
        # Get the corresponding answer and ticket type
        best_match_answer = faq_df.iloc[top_indices[0]]['Answer']
        best_match_type = faq_df.iloc[top_indices[0]]['Ticket_Type']
        
        # Format the response to be more helpful
        formatted_answer = format_response(best_match_answer, best_match_type)
        
        # Add context from similar questions if confidence is medium
        if 0.15 <= top_scores[0] < 0.3 and len(top_indices) > 1:
            additional_context = f"\n\nAlternatively, if you're asking about {faq_df.iloc[top_indices[1]]['Original_Question']}, please let me know and I can provide information on that instead."
            formatted_answer += additional_context
        
        return jsonify({
            "response": formatted_answer,
            "similarity": float(top_scores[0])
        })
        
    except Exception as e:
        print(f"Chat Error: {e}")
        return jsonify({"response": f"I apologize, but I encountered an issue while processing your question. Could you try rephrasing it?"})

@app.route('/api/model_status', methods=['GET'])
def model_status():
    """Return the current status of the model (trained or not)."""
    global model_trained
    return jsonify({"trained": model_trained})

def load_model_if_exists():
    """Load the model from disk if it exists."""
    global vectorizer, model_trained, faq_df, tfidf_matrix, ticket_types
    
    if os.path.exists('model.pkl'):
        try:
            with open('model.pkl', 'rb') as f:
                load_data = pickle.load(f)
                if len(load_data) == 4:  # New format with ticket types
                    vectorizer, faq_df, tfidf_matrix, ticket_types = load_data
                else:  # Old format compatibility
                    vectorizer, faq_df, tfidf_matrix = load_data
                    ticket_types = None
            model_trained = True
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")

if __name__ == '__main__':
    # Try to load existing model
    load_model_if_exists()
    
    # Run the Flask app
    app.run(debug=True)