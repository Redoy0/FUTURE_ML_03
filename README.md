# Customer Support Chatbot

This project creates an intelligent customer support chatbot using NLP techniques. The chatbot can answer customer queries using a provided CSV dataset and includes a web interface for user interaction.

## Project Structure

```
customer-support-chatbot/
├── app.py                        # Flask backend application
├── static/
│   ├── styles.css                # CSS styling for the web interface
│   └── script.js                 # JavaScript for frontend interactivity
├── templates/
│   └── index.html                # HTML template for the web interface
├── model.pkl                     # Saved model (generated after training)
├── customer_support_tickets.csv  # Dataset (already included)
└── requirements.txt              # Python dependencies
```

## Prerequisites

- Python 3.7+
- pip (Python package manager)
- NLTK resources (downloaded automatically during first run)

## Setup Instructions

1. **Create a project directory and virtual environment**:

```bash
# Create a project directory
mkdir customer-support-chatbot
cd customer-support-chatbot

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

2. **Install required packages**:

```bash
pip install -r requirements.txt
```

3. **Create the project structure**:

```bash
mkdir -p static templates
```

4. **Create the project files**:
   - Copy the provided code into their respective files:
     - `app.py` in the root directory
     - `index.html` in the `templates` directory
     - `styles.css` and `script.js` in the `static` directory

5. **Add the dataset**:
   - Download the Customer Support Ticket Dataset from [Kaggle](https://www.kaggle.com/datasets/waseemalastal/customer-support-ticket-dataset)
   - Place the CSV file in the project root directory as `customer_support_tickets.csv`

## Running the Application

1. **Start the Flask server**:

```bash
python app.py
```

2. **Access the application**:
   - Open your web browser and navigate to `http://127.0.0.1:5000/`

3. **Using the chatbot**:
   - Click the "Train Model" button to train the chatbot on the dataset
   - After training, start asking customer support related questions in the chat interface

## How It Works

1. **Before Training**:
   - The chatbot responds with random funny text lines when asked questions

2. **Training Process**:
   - Loads the CSV dataset
   - Preprocesses the data by combining ticket subjects and descriptions
   - Creates a TF-IDF vectorizer to convert text into numerical features
   - Saves the model for future use

3. **After Training**:
   - When a user asks a question, the chatbot:
     - Converts the question into a TF-IDF vector
     - Calculates cosine similarity with all questions in the dataset
     - Returns the answer that corresponds to the most similar question
     - If no good match is found, returns a fallback message

## Technical Implementation Details

### Backend (Flask)

- **Model**: Uses TF-IDF vectorization and cosine similarity to find the most relevant answer
- **API Endpoints**:
  - `/` - Serves the main web interface
  - `/api/train` - Handles model training
  - `/api/chat` - Processes user messages and returns bot responses
  - `/api/model_status` - Returns whether the model is trained
  - `/api/upload_csv` - Handles CSV file uploads

### Frontend (HTML/CSS/JavaScript)

- **Chat Interface**: Clean and responsive design with message bubbles
- **Features**:
  - Real-time chat experience with typing indicators
  - File upload for the dataset
  - Training status indicator
  - Scrollable message history

## Example Funny Pre-Training Responses

1. "I don't know anything yet because you haven't trained me! Click the 'Train Model' button to enlighten me."
2. "Error 404: Knowledge not found. Have you tried turning me on? I mean, training me?"
3. "My brain is as empty as a developer's wallet after buying a new MacBook. Please train me!"
4. "I'm currently as useful as a chocolate teapot. Train me to be more helpful!"
5. "Still loading personality... Please click 'Train Model' to accelerate the process."

## Extending the Project

Here are some ways you could extend this project:
- Implement more sophisticated NLP models (e.g., using BERT or other transformer models)
- Add features like sentiment analysis of user queries
- Include a feedback mechanism to improve answers over time
- Implement user authentication
- Add multi-language support

## Troubleshooting

- If the application fails to start, ensure all dependencies are installed
- If training fails, check the format of your CSV file
- If the model gives irrelevant answers, try adjusting the similarity threshold in the code