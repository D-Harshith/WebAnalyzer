let currentScores = {};

async function analyzeUrl() {
    const urlInput = document.getElementById('urlInput').value;
    const analyzeBtn = document.getElementById('analyzeBtn');
    const results = document.getElementById('results');
    const scoreGauge = document.getElementById('scoreGauge');
    const finalScore = document.getElementById('finalScore');
    const scoreDetails = document.getElementById('scoreDetails');
    const screenshot = document.getElementById('screenshot');

    if (!urlInput) {
        alert('Please enter a valid URL');
        return;
    }

    analyzeBtn.classList.add('analyzing');
    analyzeBtn.textContent = 'Analyzing...';
    analyzeBtn.disabled = true;

    try {
        const response = await fetch('/api/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ url: urlInput })
        });

        if (!response.ok) {
            throw new Error('Failed to analyze URL');
        }

        const data = await response.json();
        currentScores = data.scores;

        // Update gauge
        finalScore.textContent = data.scores['Final AI Visibility Score'];
        scoreGauge.style.background = `conic-gradient(#00ff00 0% ${data.scores['Final AI Visibility Score']}%, #333333 ${data.scores['Final AI Visibility Score']}% 100%)`;

        // Update score details
        scoreDetails.innerHTML = '';
        for (const [key, value] of Object.entries(data.scores)) {
            const detail = document.createElement('div');
            detail.className = 'bg-gray-900 p-4 rounded-lg';
            detail.innerHTML = `<strong>${key}:</strong> ${value}${key.includes('Count') ? '' : '%'}`;
            scoreDetails.appendChild(detail);
        }

        // Update screenshot
        if (data.screenshot && data.screenshot.length > 0) {
            screenshot.src = `data:image/png;base64,${data.screenshot}`;
        } else {
            screenshot.src = 'https://via.placeholder.com/1280x720?text=Screenshot+Not+Available';
            screenshot.alt = 'Screenshot not available';
        }

        results.classList.remove('hidden');
    } catch (error) {
        console.error('Error:', error);
        alert('An error occurred while analyzing the URL');
    } finally {
        analyzeBtn.classList.remove('analyzing');
        analyzeBtn.textContent = 'Analyze';
        analyzeBtn.disabled = false;
    }
}

function openChatModal() {
    document.getElementById('chatModal').style.display = 'flex';
    document.getElementById('modalOverlay').style.display = 'block';
}

function closeChatModal() {
    document.getElementById('chatModal').style.display = 'none';
    document.getElementById('modalOverlay').style.display = 'none';
}

async function sendChatQuery() {
    const chatInput = document.getElementById('chatInput');
    const chatContainer = document.getElementById('chatContainer');
    const query = chatInput.value.trim();

    if (!query) {
        alert('Please enter a question');
        return;
    }

    // Add user message
    const userMessage = document.createElement('div');
    userMessage.className = 'chat-message user-message';
    userMessage.textContent = query;
    chatContainer.appendChild(userMessage);

    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query, scores: currentScores })
        });

        if (!response.ok) {
            throw new Error('Failed to get chat response');
        }

        const data = await response.json();
        const botMessage = document.createElement('div');
        botMessage.className = 'chat-message bot-message';
        botMessage.textContent = data.response;
        chatContainer.appendChild(botMessage);
    } catch (error) {
        console.error('Error:', error);
        const botMessage = document.createElement('div');
        botMessage.className = 'chat-message bot-message';
        botMessage.textContent = 'Sorry, an error occurred while processing your request.';
        chatContainer.appendChild(botMessage);
    }

    chatInput.value = '';
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

document.getElementById('chatInput').addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        sendChatQuery();
    }
});