# Customer Support Chatbot - Project Walkthrough

This document provides a detailed explanation of the code and functionality of the Customer Support Chatbot project.

## 1. Backend (Flask Application)

### Key Components in `app.py`

#### Global Variables
```python
vectorizer = None
model_trained = False
faq_df = None
tfidf_matrix = None
funny_responses = [
    "I don't know anything yet because you haven't trained me! Click the 'Train Model' button to enlighten me.",
    "Error 404: Knowledge not found. Have you tried turning me on? I mean, training me?",
    # ... more responses
]
```
- These global variables maintain the state of the model and data across different requests.
- `funny_responses` contains the pre-training responses that the bot sends when asked questions.

#### Route: `/api/train`
```python
@app.route('/api/train', methods=['POST'])
def train_model():
    """Train the NLP model on the customer support dataset."""
    global vectorizer, model_trained, faq_df, tfidf_matrix
    
    try:
        # Load the dataset
        data = pd.read_csv('customer_support_tickets.csv')
        
        # Clean and prepare the data
        data = data.dropna(subset=['Ticket Subject', 'Ticket Description', 'Resolution'])
        
        # Combine subject and description for better context
        data['Question'] = data['Ticket Subject'] + " " + data['Ticket Description']
        data['Answer'] = data['Resolution']
        
        # Create a simplified dataset for question answering
        faq_df = data[['Question', 'Answer']].copy()
        
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(faq_df['Question'])
        
        # Save the model for future use
        with open('model.pkl', 'wb') as f:
            pickle.dump((vectorizer, faq_df, tfidf_matrix), f)
        
        model_trained = True
        return jsonify({"status": "success", "message": "Model trained successfully!"})
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})
```
- This route handles the training process.
- It loads the CSV data, combines the Ticket Subject and Description to form questions, and uses the Resolution field as answers.
- It creates a TF-IDF vectorizer to convert text questions into numerical vectors.
- The model components are saved to disk for persistence.

#### Route: `/api/chat`
```python
@app.route('/api/chat', methods=['POST'])
def chat():
    """Process user messages and return bot responses."""
    global model_trained, vectorizer, faq_df, tfidf_matrix
    
    user_message = request.json.get('message', '')
    
    if not model_trained:
        # Return a random funny response if model is not trained
        return jsonify({"response": random.choice(funny_responses)})
    
    try:
        # Transform user query using the vectorizer
        user_vector = vectorizer.transform([user_message])
        
        # Calculate similarity with all FAQs
        similarity_scores = cosine_similarity(user_vector, tfidf_matrix).flatten()
        
        # Find the most similar question
        best_match_index = similarity_scores.argmax()
        similarity_score = similarity_scores[best_match_index]
        
        # If similarity is too low, return a fallback message
        if similarity_score < 0.1:
            return jsonify({
                "response": "I'm not sure I understand. Could you please rephrase your question?"
            })
        
        # Get the corresponding answer
        best_match_answer = faq_df.iloc[best_match_index]['Answer']
        
        return jsonify({
            "response": best_match_answer,
            "similarity": float(similarity_score)
        })
        
    except Exception as e:
        return jsonify({"response": f"An error occurred: {str(e)}"})
```
- This route handles processing user messages and returning bot responses.
- If the model is not trained, it returns a random funny response.
- If the model is trained, it:
  1. Transforms the user message into a TF-IDF vector
  2. Calculates the cosine similarity between the user message and all questions in the dataset
  3. Finds the most similar question and returns its corresponding answer
  4. If no good match is found (similarity < 0.1), it returns a fallback message

#### Route: `/api/upload_csv`
```python
@app.route('/api/upload_csv', methods=['POST'])
def upload_csv():
    """Handle CSV file upload."""
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file part"})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"status": "error", "message": "No selected file"})
    
    try:
        file.save('customer_support_tickets.csv')
        return jsonify({"status": "success", "message": "File uploaded successfully"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})
```
- This route handles uploading the CSV dataset.
- It saves the uploaded file to the server for later use in training.

#### Helper Function: `load_model_if_exists`
```python
def load_model_if_exists():
    """Load the model from disk if it exists."""
    global vectorizer, model_trained, faq_df, tfidf_matrix
    
    if os.path.exists('model.pkl'):
        try:
            with open('model.pkl', 'rb') as f:
                vectorizer, faq_df, tfidf_matrix = pickle.load(f)
            model_trained = True
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
```
- This function checks if a previously trained model exists and loads it.
- This ensures that the model persists between server restarts.

## 2. Frontend (HTML/CSS/JavaScript)

### HTML Structure (`index.html`)

The HTML structure includes:
- A header with logo and model status indicator
- A message area to display the chat history
- An input area for the user to type messages
- Control buttons for uploading the dataset and training the model

Key sections:
```html
<div class="chat-messages" id="chat-messages">
    <!-- Messages will be dynamically added here -->
    <div class="message bot-message">
        <div class="message-content">
            Hello! I'm your customer support chatbot. Please train me using the button below before asking questions.
        </div>
    </div>
</div>

<div class="controls">
    <div class="dataset-controls">
        <input type="file" id="csv-upload" accept=".csv" style="display: none;">
        <button id="upload-button" class="action-button">
            <i class="fas fa-file-upload"></i> Upload Dataset
        </button>
        <span id="file-selected" class="file-name">No file selected</span>
    </div>
    <button id="train-button" class="action-button">
        <span class="button-content">
            <i class="fas fa-brain"></i> Train Model
        </span>
        <span class="loading-spinner" id="training-spinner">
            <i class="fas fa-spinner fa-spin"></i>
        </span>
    </button>
</div>
```

### CSS Styling (`styles.css`)

The CSS provides:
- A clean and modern chat interface
- Distinct styling for user and bot messages
- Status indicators for model training
- Responsive design for different screen sizes

Key styling elements:
```css
.message-content {
    max-width: 70%;
    padding: 12px 16px;
    border-radius: 18px;
    font-size: 15px;
    line-height: 1.4;
}

.bot-message .message-content {
    background-color: #f0f2f5;
    color: #333;
    border-bottom-left-radius: 5px;
}

.user-message .message-content {
    background-color: #4a6fa5;
    color: white;
    border-bottom-right-radius: 5px;
}
```

### JavaScript Functionality (`script.js`)

The JavaScript code handles:
- User interaction with the chat interface
- API calls to the backend
- Dynamic updating of the UI based on model status

Key functions:

#### `sendMessage()`
```javascript
async function sendMessage() {
    const message = userInput.value.trim();
    if (message === '') return;
    
    // Add user message to chat
    addMessage(message, 'user');
    userInput.value = '';
    
    // Simulate typing indicator
    const typingIndicator = addTypingIndicator();
    
    try {
        // Send message to server
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ message })
        });
        
        const data = await response.json();
        
        // Remove typing indicator
        removeTypingIndicator(typingIndicator);
        
        // Add bot response
        addMessage(data.response, 'bot');
        
    } catch (error) {
        console.error('Error sending message:', error);
        removeTypingIndicator(typingIndicator);
        addMessage('Sorry, there was an error processing your message.', 'bot');
    }
    
    // Scroll to bottom
    scrollToBottom();
}
```
- This function sends the user's message to the server and displays the bot's response.
- It includes a typing indicator to simulate the bot typing, enhancing the user experience.

#### `trainModel()`
```javascript
async function trainModel() {
    // Show loading state
    trainingSpinner.style.display = 'inline-block';
    trainButton.disabled = true;
    
    try {
        const response = await fetch('/api/train', {
            method: 'POST'
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            addMessage('Training completed successfully! I am now ready to answer your questions about customer support.', 'bot');
            updateModelStatus(true);
        } else {
            addMessage(`Training failed: ${data.message}`, 'bot');
        }
    } catch (error) {
        console.error('Error training model:', error);
        addMessage('Sorry, there was an error training the model.', 'bot');
    } finally {
        // Hide loading state
        trainingSpinner.style.display = 'none';
        trainButton.disabled = false;
    }
}
```
- This function sends a request to train the model and updates the UI accordingly.
- It shows a loading spinner during training and provides feedback to the user about the training status.

## 3. NLP Approach

### Term Frequency-Inverse Document Frequency (TF-IDF)

The chatbot uses TF-IDF vectorization to convert text into numerical vectors. This approach:
- Assigns weights to words based on their frequency in a document and rarity across documents
- Gives higher weights to distinctive words that are more important for document classification
- Allows for measuring the similarity between questions using cosine similarity

### Cosine Similarity

Cosine similarity measures the cosine of the angle between two vectors, providing a value between -1 and 1:
- 1 means the vectors are identical
- 0 means the vectors are orthogonal (no similarity)
- -1 means the vectors are opposite

In our context:
- User question is converted to a TF-IDF vector
- This vector is compared with all question vectors in the dataset
- The answer corresponding to the most similar question is returned

### Fallback Mechanism

If no good match is found (similarity < 0.1), the chatbot returns a fallback message:
```python
if similarity_score < 0.1:
    return jsonify({
        "response": "I'm not sure I understand. Could you please rephrase your question?"
    })
```

## 4. Testing and Evaluation

### Sample Interaction
1. User uploads the dataset
2. User clicks the "Train Model" button
3. The bot indicates it's trained successfully
4. User asks: "How do I reset my password?"
5. Bot retrieves the most similar question from the dataset and returns the corresponding resolution

### Potential Improvements

1. **Better NLP Models**:
   - Use more advanced models like BERT or other transformer-based models
   - Implement sentence embeddings for better semantic understanding

2. **User Experience**:
   - Add conversation history persistence
   - Implement more advanced typing indicators
   - Add voice input/output options

3. **Model Quality**:
   - Implement feedback mechanisms for improving answers
   - Add multiple response options for ambiguous questions
   - Implement intent classification for better understanding of user needs

4. **Additional Features**:
   - Add FAQ suggestion buttons
   - Implement automatic ticket creation for unresolved issues
   - Add sentiment analysis to detect frustrated users

## 5. Deployment Considerations

For deployment to production, consider:

1. **Security**:
   - Implement proper input validation
   - Add user authentication if needed
   - Secure API endpoints

2. **Scalability**:
   - Move to a production-ready server (e.g., Gunicorn + Nginx)
   - Consider containerization (Docker)
   - Implement caching for frequent queries

3. **Monitoring**:
   - Add logging for tracking errors and usage
   - Implement analytics to track common questions
   - Set up alerting for system issues

4. **Maintenance**:
   - Regularly update the dataset with new information
   - Periodically retrain the model to incorporate new data
   - Review and update fallback responses based on user feedback