let currentScores = null;

function openChatModal() {
    document.getElementById('modalOverlay').style.display = 'block';
    document.getElementById('chatModal').style.display = 'flex';
    document.getElementById('chatInput').focus();
}

function closeChatModal() {
    document.getElementById('modalOverlay').style.display = 'none';
    document.getElementById('chatModal').style.display = 'none';
}

async function analyzeUrl() {
    const urlInput = document.getElementById('urlInput').value;
    if (!urlInput.match(/^https?:\/\/[^\s/$.?#].[^\s]*$/)) {
        alert('Please enter a valid URL');
        return;
    }
    const analyzeBtn = document.getElementById('analyzeBtn');
    analyzeBtn.textContent = 'Analyzing...';
    analyzeBtn.disabled = true;
    analyzeBtn.classList.add('analyzing');
    try {
        const response = await fetch('/api/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ url: urlInput })
        });
        console.log('Response status:', response.status);
        const text = await response.text();
        console.log('Response text:', text);
        if (!response.ok) throw new Error(`HTTP error: ${response.status}`);
        const data = JSON.parse(text);
        if (!data.scores || !data.scores['Final AI Visibility Score']) {
            throw new Error('Invalid response: scores or Final AI Visibility Score missing');
        }
        currentScores = data.scores;
        displayResults(data.scores, data.screenshot);
        document.getElementById('results').classList.remove('hidden');
    } catch (error) {
        console.error('Error:', error);
        alert('Error analyzing URL: ' + error.message);
    } finally {
        analyzeBtn.textContent = 'Analyze';
        analyzeBtn.disabled = false;
        analyzeBtn.classList.remove('analyzing');
    }
}

function displayResults(scores, screenshot) {
    const finalScore = scores['Final AI Visibility Score'];
    const scoreGauge = document.getElementById('scoreGauge');
    const finalScoreSpan = document.getElementById('finalScore');
    scoreGauge.style.background = `conic-gradient(#00ff00 0% 0%, #333333 0% 100%)`;
    finalScoreSpan.textContent = '0';

    // Animate the gauge
    let currentScore = 0;
    const increment = finalScore / 50;
    const animateGauge = setInterval(() => {
        currentScore += increment;
        if (currentScore >= finalScore) {
            currentScore = finalScore;
            clearInterval(animateGauge);
        }
        finalScoreSpan.textContent = currentScore.toFixed(1);
        scoreGauge.style.background = `conic-gradient(#00ff00 0% ${currentScore}%, #333333 ${currentScore}% 100%)`;
    }, 20);

    const scoreDetails = document.getElementById('scoreDetails');
    scoreDetails.innerHTML = '';
    for (const [key, value] of Object.entries(scores)) {
        if (key !== 'Final AI Visibility Score' && !key.includes('Count')) {
            const div = document.createElement('div');
            div.className = 'bg-gray-900 p-4 rounded-lg';
            div.innerHTML = `<p class="font-semibold">${key}</p><p class="text-2xl">${value}%</p>`;
            scoreDetails.appendChild(div);
        }
    }

    const screenshotImg = document.getElementById('screenshot');
    screenshotImg.src = `data:image/png;base64,${screenshot}`;
}

function addChatMessage(text, isUser) {
    const chatContainer = document.getElementById('chatContainer');
    const messageDiv = document.createElement('div');
    messageDiv.className = `chat-message ${isUser ? 'user-message' : 'bot-message'}`;
    messageDiv.textContent = text;
    chatContainer.appendChild(messageDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

async function sendChatQuery() {
    const chatInput = document.getElementById('chatInput');
    const query = chatInput.value.trim();
    if (!query) {
        alert('Please enter a question');
        return;
    }

    const chatBtn = document.getElementById('chatBtn');
    chatBtn.textContent = 'Sending...';
    chatBtn.disabled = true;

    addChatMessage(query, true);
    chatInput.value = '';

    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query, scores: currentScores })
        });
        if (!response.ok) throw new Error(`HTTP error: ${response.status}`);
        const data = await response.json();
        addChatMessage(data.response, false);
    } catch (error) {
        addChatMessage('Error: ' + error.message, false);
    } finally {
        chatBtn.textContent = 'Send';
        chatBtn.disabled = false;
    }
}