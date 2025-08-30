let currentScores = {};

async function analyzeUrl() {
    const urlInput = document.getElementById('urlInput').value;
    const analyzeBtn = document.getElementById('analyzeBtn');
    const results = document.getElementById('results');
    const scoreGauge = document.getElementById('scoreGauge');
    const finalScore = document.getElementById('finalScore');
    const scoreMessage = document.getElementById('scoreMessage');
    const scoreDetails = document.getElementById('scoreDetails');
    const screenshot = document.getElementById('screenshot');

    if (!urlInput) {
        alert('Please enter a valid URL');
        return;
    }

    // Clear previous results and hide the section
    results.classList.add('hidden');
    scoreDetails.innerHTML = '';
    screenshot.src = '';
    scoreMessage.textContent = 'AI Visibility Score';
    finalScore.textContent = '0';
    scoreGauge.style.background = `conic-gradient(#00ff00 0% 0%, #333333 0% 100%)`;

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
            const errorText = await response.text();
            if (errorText.includes('Crawling blocked by robots.txt')) {
                throw new Error('Crawling blocked by robots.txt');
            }
            throw new Error('Failed to analyze URL');
        }

        const data = await response.json();
        currentScores = data.scores;

        // Update gauge
        finalScore.textContent = data.scores['Final AI Visibility Score'];
        scoreGauge.style.background = `conic-gradient(#00ff00 0% ${data.scores['Final AI Visibility Score']}%, #333333 ${data.scores['Final AI Visibility Score']}% 100%)`;

        // Update score message based on final score
        const finalScoreValue = data.scores['Final AI Visibility Score'];
        if (finalScoreValue > 75) {
            scoreMessage.textContent = 'Congratulations! We can further improve your score.';
        } else if (finalScoreValue > 50) {
            scoreMessage.textContent = 'A lot of improvement is needed.';
        } else {
            scoreMessage.textContent = 'We can definitely improve your score to boost visibility.';
        }

        // Update score details
        scoreDetails.innerHTML = '';
        for (const [key, value] of Object.entries(data.scores)) {
            const detail = document.createElement('div');
            detail.className = 'bg-gray-900 p-4 rounded-lg';
            detail.innerHTML = `<strong>${key}:</strong> ${typeof value === 'number' && key !== 'Total Heading Count' ? value + '%' : value}`;
            scoreDetails.appendChild(detail);
        }

        // Display mentions in full-width
        const mentions = document.createElement('div');
        mentions.className = 'bg-gray-900 p-4 rounded-lg mt-4 col-span-1 md:col-span-2';
        mentions.innerHTML = `<strong>LLM Mentions:</strong> ${data.total_mentions} across topics: ${Object.keys(data.mentions).join(', ')}`;
        scoreDetails.appendChild(mentions);

        // Display suggestions in full-width if available, formatted with new lines and removing "0. --------------------------" patterns
        if (data.suggestions) {
            const formattedSuggestions = data.suggestions
                .replace(/\*\*/g, '') // Remove bold markers
                .replace(/(\d+\. )[^\n]+?0\. --------------------------/g, '$1') // Remove "0. --------------------------" patterns
                .replace(/(\d+\. )/g, '<br>$1'); // Ensure each point starts on a new line
            const suggestions = document.createElement('div');
            suggestions.className = 'bg-gray-900 p-4 rounded-lg mt-4 col-span-1 md:col-span-2';
            suggestions.innerHTML = `<strong>Suggestions to Improve AI Visibility:</strong> ${formattedSuggestions}`;
            scoreDetails.appendChild(suggestions);
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
        if (error.message === 'Crawling blocked by robots.txt') {
            alert('Unable to crawl the URL. This site has blocked crawling via robots.txt. Please check the robots.txt file or try a different URL.');
        } else {
            alert('An error occurred while analyzing the URL');
        }
    } finally {
        analyzeBtn.classList.remove('analyzing');
        analyzeBtn.textContent = 'Analyze';
        analyzeBtn.disabled = false;
    }
}

function openChatModal() {
    const chatModal = document.getElementById('chatModal');
    const modalOverlay = document.getElementById('modalOverlay');
    const chatbotIcon = document.getElementById('chatbotIcon');

    // Hide chatbot icon when modal is open
    chatbotIcon.style.display = 'none';
    chatModal.style.display = 'flex';
    modalOverlay.style.display = 'block';

    const chatContainer = document.getElementById('chatContainer');
    if (!chatContainer.querySelector('.default-message')) {
        const defaultMessage = document.createElement('div');
        defaultMessage.className = 'chat-message bot-message default-message';
        defaultMessage.textContent = "This website assesses your site's online visibility and allows you to discuss your scores after analysis.";
        chatContainer.appendChild(defaultMessage);
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }
}

function closeChatModal() {
    const chatModal = document.getElementById('chatModal');
    const modalOverlay = document.getElementById('modalOverlay');
    const chatbotIcon = document.getElementById('chatbotIcon');

    // Show chatbot icon when modal is closed
    chatbotIcon.style.display = 'flex';
    chatModal.style.display = 'none';
    modalOverlay.style.display = 'none';
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

// Ensure chatbot icon is visible on page load
window.addEventListener('load', () => {
    document.getElementById('chatbotIcon').style.display = 'flex';
});