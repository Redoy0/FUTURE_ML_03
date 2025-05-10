// DOM Elements
const chatMessages = document.getElementById('chat-messages');
const userInput = document.getElementById('user-input');
const sendButton = document.getElementById('send-button');
const trainButton = document.getElementById('train-button');
const modelStatus = document.getElementById('model-status');
const statusText = document.getElementById('status-text');
const trainingSpinner = document.getElementById('training-spinner');

// State variables
let modelTrained = false;

// Check model status on page load
document.addEventListener('DOMContentLoaded', async () => {
    try {
        const response = await fetch('/api/model_status');
        const data = await response.json();
        
        if (data.trained) {
            updateModelStatus(true);
            addMessage("The model is already trained and ready to assist you with customer support queries!", 'bot');
        } else {
            addMessage("Welcome! Please click the 'Train Model' button to start. The dataset is already loaded.", 'bot');
        }
    } catch (error) {
        console.error('Error checking model status:', error);
    }
});

// Event Listeners
sendButton.addEventListener('click', sendMessage);
userInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        sendMessage();
    }
});

trainButton.addEventListener('click', trainModel);

// Functions
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

function addMessage(message, sender) {
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message', `${sender}-message`);
    
    const contentDiv = document.createElement('div');
    contentDiv.classList.add('message-content');
    contentDiv.textContent = message;
    
    messageDiv.appendChild(contentDiv);
    chatMessages.appendChild(messageDiv);
    
    scrollToBottom();
}

function addTypingIndicator() {
    const typingDiv = document.createElement('div');
    typingDiv.classList.add('message', 'bot-message', 'typing-indicator');
    
    const contentDiv = document.createElement('div');
    contentDiv.classList.add('message-content');
    contentDiv.innerHTML = '<div class="typing-dots"><span>.</span><span>.</span><span>.</span></div>';
    
    typingDiv.appendChild(contentDiv);
    chatMessages.appendChild(typingDiv);
    
    scrollToBottom();
    return typingDiv;
}

function removeTypingIndicator(indicator) {
    if (indicator && indicator.parentNode) {
        indicator.parentNode.removeChild(indicator);
    }
}

function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

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

// This function has been removed as we're no longer handling file uploads from the frontend

function updateModelStatus(trained) {
    modelTrained = trained;
    
    if (trained) {
        modelStatus.classList.remove('untrained');
        modelStatus.classList.add('trained');
        statusText.textContent = 'Trained';
    } else {
        modelStatus.classList.remove('trained');
        modelStatus.classList.add('untrained');
        statusText.textContent = 'Untrained';
    }
}