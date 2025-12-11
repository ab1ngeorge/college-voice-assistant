// Configuration
const API_BASE_URL = 'http://localhost:8000';
let currentLanguage = 'auto';
let isRecording = false;
let mediaRecorder;
let audioChunks = [];
let recognition = null;

// DOM Elements
const messageInput = document.getElementById('messageInput');
const sendBtn = document.getElementById('sendBtn');
const voiceBtn = document.getElementById('voiceBtn');
const voiceStatus = document.getElementById('voiceStatus');
const chatMessages = document.getElementById('chatMessages');
const langButtons = document.querySelectorAll('.lang-btn');
const exampleButtons = document.querySelectorAll('.example-btn');
const adminPanelBtn = document.getElementById('adminPanelBtn');
const adminModal = document.getElementById('adminModal');
const closeModal = document.getElementById('closeModal');
const addDocumentForm = document.getElementById('addDocumentForm');
const documentsList = document.getElementById('documentsList');
const logsContainer = document.getElementById('logsContainer');
const responseAudio = document.getElementById('responseAudio');
const tabButtons = document.querySelectorAll('.tab-btn');
const tabContents = document.querySelectorAll('.tab-content');

// Initialize Speech Recognition
function initSpeechRecognition() {
    if ('webkitSpeechRecognition' in window) {
        recognition = new webkitSpeechRecognition();
        recognition.continuous = false;
        recognition.interimResults = false;
        
        // Set language based on selection
        updateRecognitionLanguage();
        
        recognition.onstart = () => {
            isRecording = true;
            voiceBtn.classList.add('listening');
            voiceStatus.style.display = 'flex';
            addMessage('system', 'Listening... Speak now!');
        };
        
        recognition.onresult = (event) => {
            const transcript = event.results[0][0].transcript;
            messageInput.value = transcript;
            
            // Auto-send after recognition
            setTimeout(() => sendMessage(), 500);
        };
        
        recognition.onerror = (event) => {
            console.error('Speech recognition error:', event.error);
            addMessage('system', `Speech recognition error: ${event.error}`);
        };
        
        recognition.onend = () => {
            isRecording = false;
            voiceBtn.classList.remove('listening');
            voiceStatus.style.display = 'none';
        };
    } else {
        console.warn('Speech recognition not supported');
        voiceBtn.disabled = true;
        voiceBtn.innerHTML = '<i class="fas fa-microphone-slash"></i>';
        voiceBtn.title = 'Speech recognition not supported';
    }
}

// Update recognition language based on selection
function updateRecognitionLanguage() {
    if (!recognition) return;
    
    switch(currentLanguage) {
        case 'ml':
            recognition.lang = 'ml-IN';
            break;
        case 'manglish':
            recognition.lang = 'en-IN';
            break;
        default:
            recognition.lang = 'en-US';
    }
}

// Add message to chat
function addMessage(sender, text, isExample = false) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}`;
    
    const time = new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
    
    const senderName = sender === 'user' ? 'You' : 'Assistant';
    
    messageDiv.innerHTML = `
        <div class="message-content">
            <strong>${senderName}:</strong> ${text}
        </div>
        <div class="message-time">${time}</div>
    `;
    
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    // Add Malayalam font class if needed
    if (text.match(/[\u0D00-\u0D7F]/)) {
        messageDiv.classList.add('malayalam-text');
    }
}

// Send message to backend
async function sendMessage() {
    const text = messageInput.value.trim();
    if (!text) return;
    
    // Add user message
    addMessage('user', text);
    messageInput.value = '';
    
    try {
        const response = await fetch(`${API_BASE_URL}/api/query`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                text: text,
                language: currentLanguage === 'auto' ? null : currentLanguage
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        // Add bot message
        addMessage('bot', data.answer);
        
        // Speak response if audio URL is available
        if (data.audio_url) {
            responseAudio.src = `${API_BASE_URL}${data.audio_url}`;
            responseAudio.play().catch(e => console.log('Audio play failed:', e));
        } else {
            // Use browser TTS as fallback
            speakText(data.answer, data.language);
        }
        
        // Log interaction
        logToSystem(`User query: "${text}" -> Response: "${data.answer.substring(0, 50)}..."`);
        
    } catch (error) {
        console.error('Error sending message:', error);
        addMessage('bot', 'Sorry, there was an error processing your request. Please try again.');
        logToSystem(`Error: ${error.message}`, 'error');
    }
}

// Speak text using browser TTS
function speakText(text, language) {
    if ('speechSynthesis' in window) {
        const utterance = new SpeechSynthesisUtterance(text);
        
        // Set language
        switch(language) {
            case 'ml':
                utterance.lang = 'ml-IN';
                break;
            case 'manglish':
                utterance.lang = 'en-IN';
                break;
            default:
                utterance.lang = 'en-US';
        }
        
        utterance.rate = 1.0;
        utterance.pitch = 1.0;
        utterance.volume = 1.0;
        
        speechSynthesis.speak(utterance);
    }
}

// Start/stop voice recording
function toggleVoiceRecording() {
    if (!recognition) {
        addMessage('system', 'Voice recognition is not supported in your browser.');
        return;
    }
    
    if (isRecording) {
        recognition.stop();
    } else {
        recognition.start();
    }
}

// Load documents from backend
async function loadDocuments() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/documents`);
        const data = await response.json();
        
        documentsList.innerHTML = '';
        data.documents.forEach((doc, index) => {
            const docDiv = document.createElement('div');
            docDiv.className = 'document-item';
            
            const textParts = doc.text.split('|').map(p => p.trim());
            docDiv.innerHTML = `
                <div><strong>Document #${index + 1}</strong></div>
                <div><span class="lang">EN</span> ${textParts[0] || ''}</div>
                ${textParts[1] ? `<div><span class="lang">ML</span> ${textParts[1]}</div>` : ''}
                ${textParts[2] ? `<div><span class="lang">Manglish</span> ${textParts[2]}</div>` : ''}
            `;
            
            documentsList.appendChild(docDiv);
        });
        
        logToSystem(`Loaded ${data.documents.length} documents from knowledge base`);
        
    } catch (error) {
        console.error('Error loading documents:', error);
        logToSystem(`Error loading documents: ${error.message}`, 'error');
    }
}

// Add new document
async function addDocument(english, malayalam, manglish) {
    try {
        const response = await fetch(`${API_BASE_URL}/api/add-document`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                english: english,
                malayalam: malayalam,
                manglish: manglish || ""
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        addMessage('system', `✅ Document added successfully! ID: ${data.id}`);
        logToSystem(`Added new document: ${english.substring(0, 50)}...`);
        
        // Reload documents
        loadDocuments();
        
        return true;
        
    } catch (error) {
        console.error('Error adding document:', error);
        addMessage('system', `❌ Error adding document: ${error.message}`);
        logToSystem(`Error adding document: ${error.message}`, 'error');
        return false;
    }
}

// System logging
function logToSystem(message, type = 'info') {
    const logEntry = document.createElement('div');
    logEntry.className = `log-entry ${type}`;
    
    const timestamp = new Date().toLocaleTimeString();
    logEntry.textContent = `[${timestamp}] ${message}`;
    
    logsContainer.appendChild(logEntry);
    logsContainer.scrollTop = logsContainer.scrollHeight;
}

// Event Listeners
document.addEventListener('DOMContentLoaded', () => {
    // Initialize speech recognition
    initSpeechRecognition();
    
    // Load initial documents
    loadDocuments();
    
    // Test backend connection
    fetch(`${API_BASE_URL}/`)
        .then(response => response.json())
        .then(data => {
            logToSystem(`Backend connected: ${data.message}`);
        })
        .catch(error => {
            logToSystem(`Backend connection failed: ${error.message}`, 'error');
        });
});

// Language selection
langButtons.forEach(btn => {
    btn.addEventListener('click', () => {
        langButtons.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        currentLanguage = btn.dataset.lang;
        updateRecognitionLanguage();
        
        // Update input placeholder based on language
        switch(currentLanguage) {
            case 'ml':
                messageInput.placeholder = 'ഇവിടെ നിങ്ങളുടെ ചോദ്യം ടൈപ്പ് ചെയ്യുക... അല്ലെങ്കിൽ മൈക്കിൽ ക്ലിക്ക് ചെയ്ത് സംസാരിക്കുക';
                break;
            case 'manglish':
                messageInput.placeholder = 'Type your question here... or click mic to speak in Manglish';
                break;
            default:
                messageInput.placeholder = 'Type your question here... or click the mic to speak';
        }
        
        addMessage('system', `Language set to: ${btn.textContent}`);
    });
});

// Send message on button click
sendBtn.addEventListener('click', sendMessage);

// Send message on Enter key
messageInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

// Voice recording
voiceBtn.addEventListener('click', toggleVoiceRecording);

// Example questions
exampleButtons.forEach(btn => {
    btn.addEventListener('click', () => {
        messageInput.value = btn.dataset.question;
        sendMessage();
    });
});

// Admin panel
adminPanelBtn.addEventListener('click', () => {
    adminModal.style.display = 'flex';
});

closeModal.addEventListener('click', () => {
    adminModal.style.display = 'none';
});

// Close modal when clicking outside
window.addEventListener('click', (e) => {
    if (e.target === adminModal) {
        adminModal.style.display = 'none';
    }
});

// Tab switching
tabButtons.forEach(btn => {
    btn.addEventListener('click', () => {
        const tabId = btn.dataset.tab;
        
        // Update active tab button
        tabButtons.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        
        // Show corresponding tab content
        tabContents.forEach(content => {
            content.classList.remove('active');
            if (content.id === `${tabId}-tab`) {
                content.classList.add('active');
            }
        });
        
        // Load data if needed
        if (tabId === 'view-docs') {
            loadDocuments();
        }
    });
});

// Add document form submission
addDocumentForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const english = document.getElementById('englishText').value.trim();
    const malayalam = document.getElementById('malayalamText').value.trim();
    const manglish = document.getElementById('manglishText').value.trim();
    
    if (!english || !malayalam) {
        alert('Please fill in at least English and Malayalam versions');
        return;
    }
    
    const success = await addDocument(english, malayalam, manglish);
    
    if (success) {
        addDocumentForm.reset();
    }
});

// Footer links
document.getElementById('privacyLink').addEventListener('click', (e) => {
    e.preventDefault();
    addMessage('system', 'Privacy: All data is processed locally. Voice recordings are not stored.');
});

document.getElementById('aboutLink').addEventListener('click', (e) => {
    e.preventDefault();
    addMessage('system', 'College Voice Assistant v1.0 - Powered by Google Gemini AI with RAG architecture. Supports Malayalam, English & Manglish.');
});

document.getElementById('docsLink').addEventListener('click', (e) => {
    e.preventDefault();
    adminModal.style.display = 'flex';
    document.querySelector('[data-tab="view-docs"]').click();
});

// Real-time WebSocket connection for voice (optional)
function initWebSocket() {
    try {
        const ws = new WebSocket(`ws://${window.location.hostname}:8000/ws/voice`);
        
        ws.onopen = () => {
            logToSystem('WebSocket connected for real-time voice');
        };
        
        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            addMessage('bot', data.answer);
            
            // Play audio if available
            if (data.audio_path) {
                responseAudio.src = `${API_BASE_URL}/api/audio/${data.audio_path}`;
                responseAudio.play();
            }
        };
        
        ws.onerror = (error) => {
            console.error('WebSocket error:', error);
        };
        
        ws.onclose = () => {
            logToSystem('WebSocket disconnected');
        };
        
    } catch (error) {
        console.log('WebSocket not available:', error);
    }
}

// Initialize WebSocket if needed
// initWebSocket();

// Voice recording with MediaRecorder (alternative method)
async function startMediaRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];
        
        mediaRecorder.ondataavailable = (event) => {
            audioChunks.push(event.data);
        };
        
        mediaRecorder.onstop = async () => {
            const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
            await sendAudioToBackend(audioBlob);
        };
        
        mediaRecorder.start();
        isRecording = true;
        voiceBtn.classList.add('listening');
        voiceStatus.style.display = 'flex';
        
    } catch (error) {
        console.error('Error accessing microphone:', error);
        addMessage('system', 'Microphone access denied. Please allow microphone permissions.');
    }
}

async function sendAudioToBackend(audioBlob) {
    const formData = new FormData();
    formData.append('audio', audioBlob, 'recording.wav');
    
    try {
        const response = await fetch(`${API_BASE_URL}/api/voice-query`, {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        addMessage('user', data.text);
        addMessage('bot', data.answer);
        
        if (data.audio_url) {
            responseAudio.src = `${API_BASE_URL}${data.audio_url}`;
            responseAudio.play();
        }
        
    } catch (error) {
        console.error('Error sending audio:', error);
        addMessage('bot', 'Error processing voice input');
    }
}