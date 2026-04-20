document.addEventListener('DOMContentLoaded', () => {
    const analyzeBtn = document.getElementById('analyzeBtn');
    const tweetInput = document.getElementById('tweetInput');
    const resultSection = document.getElementById('resultSection');

    analyzeBtn.addEventListener('click', analyzeSentiment);

    async function analyzeSentiment() {
        const tweet = tweetInput.value.trim();
        if (!tweet) {
            alert("Please enter a tweet!");
            return;
        }

        // Show loading (optional)
        analyzeBtn.textContent = "Analyzing...";
        analyzeBtn.disabled = true;

        try {
            const response = await fetch('http://127.0.0.1:5000/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ tweet: tweet })
            });

            const result = await response.json();

            resultSection.style.display = 'block';

            if (result.sentiment === "Positive") {
                resultSection.className = 'result-section positive';
                document.getElementById('emoji').textContent = '😊';
                document.getElementById('sentiment').textContent = 'Positive';
                document.getElementById('sentiment').style.color = '#10b981';
            } else {
                resultSection.className = 'result-section negative';
                document.getElementById('emoji').textContent = '😢';
                document.getElementById('sentiment').textContent = 'Negative';
                document.getElementById('sentiment').style.color = '#ef4444';
            }

            // Star rating
            let starsHTML = '';
            for (let i = 1; i <= 5; i++) {
                starsHTML += i <= result.stars ? '⭐' : '☆';
            }
            document.getElementById('starRating').innerHTML = starsHTML;

            document.getElementById('confidence').textContent = `Confidence: ${result.confidence}%`;
            document.getElementById('tweetPreview').innerHTML = `"${result.tweet}"`;

        } catch (error) {
            alert("Error connecting to backend. Make sure app.py is running!");
            console.error(error);
        }

        // Reset button
        analyzeBtn.textContent = "Analyze Sentiment";
        analyzeBtn.disabled = false;
    }

    tweetInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            analyzeSentiment();
        }
    });
});