from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords', quiet=True)

app = Flask(__name__)
CORS(app)   # This fixes the connection issue

# Load model
try:
    model = joblib.load('twitter_sentiment_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    print("✅ Model loaded successfully!")
except Exception as e:
    print(" Error loading model:", e)

def clean_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    
    stemmer = PorterStemmer()
    text = ' '.join([stemmer.stem(word) for word in text.split()])
    return text

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        tweet = data.get('tweet', '').strip()
        
        if not tweet:
            return jsonify({'error': 'No tweet provided'}), 400
        
        cleaned_tweet = clean_text(tweet)
        vectorized = vectorizer.transform([cleaned_tweet])
        
<<<<<<< HEAD
=======
        # Get prediction from your ML model
>>>>>>> 04662a78ad04b948a9158195e7fcaf3bc91be077
        ml_prediction = model.predict(vectorized)[0]
        probability = model.predict_proba(vectorized)[0]
        ml_confidence = float(max(probability) * 100)

<<<<<<< HEAD
        lower_tweet = tweet.lower()

        # ================== STRONGER NEGATIVE DETECTION ==================
        strong_negative = ['hate', 'terrible', 'awful', 'worst', 'horrible', 'disgusting', 
                          'pathetic', 'useless', 'stupid', 'idiot', 'waste', 'sucks', 
                          'ruined', 'angry', 'frustrated', 'bad', 'never', 'shit', 'fuck']

        strong_positive = ['love', 'amazing', 'awesome', 'fantastic', 'excellent', 'best', 
                          'wonderful', 'perfect', 'happy', 'great']

        neg_count = sum(1 for word in strong_negative if word in lower_tweet)
        pos_count = sum(1 for word in strong_positive if word in lower_tweet)

        # Priority to negative if any strong negative word is present
        if neg_count >= 1:
            sentiment = "Negative"
            confidence = max(85, 95 if neg_count > 1 else 82)   # High confidence for negative
        elif pos_count >= 2:
            sentiment = "Positive"
            confidence = max(80, ml_confidence)
        elif "not " in lower_tweet and any(word in lower_tweet for word in ['good', 'great', 'like']):
            sentiment = "Negative"
            confidence = 88
        else:
            # Only trust model if no strong keywords
            sentiment = "Positive" if ml_prediction == 1 else "Negative"
            confidence = ml_confidence

=======
        # --- Strong Rule-based Boost (This will fix the bias) ---
        lower_tweet = tweet.lower()
        
        positive_keywords = ['love', 'loving', 'great', 'amazing', 'awesome', 'fantastic', 'excellent', 
                           'good', 'nice', 'best', 'wonderful', 'happy', 'perfect', 'brilliant']
        
        negative_keywords = ['hate', 'bad', 'terrible', 'awful', 'worst', 'horrible', 'disappointed', 
                           'sucks', 'stupid', 'useless']
        
        pos_count = sum(1 for word in positive_keywords if word in lower_tweet)
        neg_count = sum(1 for word in negative_keywords if word in lower_tweet)

        if pos_count > neg_count:
            sentiment = "Positive"
            confidence = max(75, ml_confidence)   # boost confidence
        elif neg_count > pos_count:
            sentiment = "Negative"
            confidence = max(75, ml_confidence)
        else:
            # If no strong keywords, use ML model result
            sentiment = "Positive" if ml_prediction == 1 else "Negative"
            confidence = ml_confidence

        # Final star rating
>>>>>>> 04662a78ad04b948a9158195e7fcaf3bc91be077
        stars = round((confidence / 100) * 5)

        return jsonify({
            'sentiment': sentiment,
            'confidence': round(confidence, 1),
            'stars': stars,
            'tweet': tweet
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)