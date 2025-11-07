from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
from textblob import TextBlob
from nrclex import NRCLex
import nltk
import random
from collections import Counter
import os

app = Flask(__name__)
CORS(app)

def download_nltk_data():
    """Download required NLTK data packages"""
    package_paths = {
        'punkt': 'tokenizers/punkt',
        'brown': 'corpora/brown',
        'vader_lexicon': 'sentiment/vader_lexicon'
    }
    for package, path in package_paths.items():
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.download(package, quiet=True)
            except:
                pass

PRODUCT_PROFILES = {
    'PixelBuds Pro': {
        'good_features': ['noise_cancellation', 'comfort', 'sound_quality', 'battery'],
        'bad_features': ['connectivity', 'price']
    },
    'Sony WH-1000XM5': {
        'good_features': ['noise_cancellation', 'battery', 'sound_quality', 'comfort'],
        'bad_features': ['price', 'design']
    },
    'AirPods Pro': {
        'good_features': ['design', 'comfort', 'integration', 'noise_cancellation'],
        'bad_features': ['price', 'battery']
    },
    'Bose QuietComfort': {
        'good_features': ['comfort', 'noise_cancellation', 'sound_quality'],
        'bad_features': ['battery', 'connectivity', 'price']
    },
    'Samsung Galaxy Buds': {
        'good_features': ['price', 'battery', 'sound_quality'],
        'bad_features': ['noise_cancellation', 'comfort']
    }
}

FEATURE_SNIPPETS = {
    'noise_cancellation': {
        'positive': [
            'Amazing noise cancellation blocks out all background noise',
            'The ANC is phenomenal, I can focus anywhere',
            'Best noise cancelling I\'ve ever experienced',
            'Incredible at silencing the world around me'
        ],
        'negative': [
            'Noise cancellation barely works, disappointing',
            'ANC is weak compared to competitors',
            'Expected better noise isolation for the price',
            'Background noise still gets through easily'
        ]
    },
    'comfort': {
        'positive': [
            'So comfortable I forget I\'m wearing them',
            'Can wear all day without any discomfort',
            'Perfect fit, very ergonomic design',
            'Incredibly lightweight and comfortable'
        ],
        'negative': [
            'Uncomfortable after an hour of use',
            'Ear pain after extended wearing',
            'Fit is awkward and keeps falling out',
            'Too tight, causes headaches'
        ]
    },
    'sound_quality': {
        'positive': [
            'Crystal clear audio with rich bass',
            'Sound quality is absolutely stunning',
            'Crisp highs and deep lows, perfect balance',
            'Audio quality exceeds all expectations'
        ],
        'negative': [
            'Sound is flat and uninspiring',
            'Audio quality is mediocre at best',
            'Bass is muddy, highs are too sharp',
            'Disappointing sound for the price point'
        ]
    },
    'battery': {
        'positive': [
            'Battery lasts all day and then some',
            'Exceptional battery life, charges quickly',
            'Never worry about running out of power',
            'Impressive 30+ hour battery life'
        ],
        'negative': [
            'Battery drains way too fast',
            'Only lasts a few hours before needing charge',
            'Battery life is abysmal',
            'Dies quickly, very inconvenient'
        ]
    },
    'connectivity': {
        'positive': [
            'Seamless Bluetooth connection every time',
            'Pairs instantly with all my devices',
            'Rock solid wireless connection',
            'Never drops connection, very reliable'
        ],
        'negative': [
            'Constant connection drops and stuttering',
            'Bluetooth pairing is a nightmare',
            'Connection is unstable and frustrating',
            'Keeps disconnecting randomly'
        ]
    },
    'price': {
        'positive': [
            'Great value for money',
            'Worth every penny, excellent investment',
            'Affordable without compromising quality',
            'Best bang for your buck'
        ],
        'negative': [
            'Overpriced for what you get',
            'Too expensive, not worth the cost',
            'Price is ridiculous for these features',
            'Way too pricey compared to alternatives'
        ]
    },
    'design': {
        'positive': [
            'Sleek and modern design, looks premium',
            'Beautiful aesthetics, very stylish',
            'Love the minimalist elegant design',
            'Gorgeous product, great build quality'
        ],
        'negative': [
            'Design feels cheap and plasticky',
            'Ugly and bulky appearance',
            'Looks dated and unattractive',
            'Build quality feels flimsy'
        ]
    },
    'integration': {
        'positive': [
            'Works perfectly with my Apple ecosystem',
            'Seamless integration across all devices',
            'Easy setup and great app support',
            'Excellent software integration'
        ],
        'negative': [
            'Poor app support and integration',
            'Doesn\'t work well with my devices',
            'Limited compatibility is frustrating',
            'App is buggy and unreliable'
        ]
    }
}

COUNTRIES = ['USA', 'GBR', 'DEU', 'FRA', 'JPN', 'IND', 'CAN', 'AUS', 'BRA', 'CHN', 
             'KOR', 'ITA', 'ESP', 'MEX', 'NLD', 'SWE', 'NOR', 'DNK', 'SGP', 'NZL']

PRODUCT_RECOMMENDATIONS = {
    'PixelBuds Pro': ['Sony WH-1000XM5', 'AirPods Pro'],
    'Sony WH-1000XM5': ['Bose QuietComfort', 'AirPods Pro'],
    'AirPods Pro': ['PixelBuds Pro', 'Sony WH-1000XM5'],
    'Bose QuietComfort': ['Sony WH-1000XM5', 'PixelBuds Pro'],
    'Samsung Galaxy Buds': ['PixelBuds Pro', 'AirPods Pro']
}

def generate_review(product_name, profile):
    """Generate a single synthetic review"""
    is_positive = random.random() > 0.3
    
    if is_positive:
        features = random.sample(profile['good_features'], min(2, len(profile['good_features'])))
        snippets = [random.choice(FEATURE_SNIPPETS[f]['positive']) for f in features]
    else:
        features = random.sample(profile['bad_features'], min(2, len(profile['bad_features'])))
        snippets = [random.choice(FEATURE_SNIPPETS[f]['negative']) for f in features]
    
    review_text = ' '.join(snippets)
    country = random.choice(COUNTRIES)
    
    return {
        'product': product_name,
        'review': review_text,
        'country': country,
        'features_mentioned': features
    }

def generate_all_reviews(num_reviews=5000):
    """Generate all synthetic reviews"""
    reviews = []
    products = list(PRODUCT_PROFILES.keys())
    
    for _ in range(num_reviews):
        product = random.choice(products)
        review = generate_review(product, PRODUCT_PROFILES[product])
        reviews.append(review)
    
    return reviews

def analyze_sentiment(text):
    """Analyze sentiment using TextBlob"""
    blob = TextBlob(text)
    return blob.sentiment.polarity

def analyze_emotions(text):
    """Analyze emotions using NRCLex"""
    emotion = NRCLex(text)
    return emotion.raw_emotion_scores

def process_reviews(reviews):
    """Process all reviews with sentiment and emotion analysis"""
    df = pd.DataFrame(reviews)
    
    df['sentiment_score'] = df['review'].apply(analyze_sentiment)
    
    emotion_data = df['review'].apply(analyze_emotions)
    
    df['joy'] = emotion_data.apply(lambda x: x.get('joy', 0))
    df['anger'] = emotion_data.apply(lambda x: x.get('anger', 0))
    df['sadness'] = emotion_data.apply(lambda x: x.get('sadness', 0))
    df['fear'] = emotion_data.apply(lambda x: x.get('fear', 0))
    df['surprise'] = emotion_data.apply(lambda x: x.get('surprise', 0))
    df['trust'] = emotion_data.apply(lambda x: x.get('trust', 0))
    df['anticipation'] = emotion_data.apply(lambda x: x.get('anticipation', 0))
    df['disgust'] = emotion_data.apply(lambda x: x.get('disgust', 0))
    
    df['sentiment_category'] = df['sentiment_score'].apply(
        lambda x: 'Positive' if x > 0.1 else ('Negative' if x < -0.1 else 'Neutral')
    )
    
    return df

print("Downloading NLTK data...")
download_nltk_data()

print("Generating synthetic reviews...")
reviews = generate_all_reviews(5000)

print("Processing reviews with sentiment and emotion analysis...")
df_reviews = process_reviews(reviews)

print(f"Generated and processed {len(df_reviews)} reviews!")

@app.route('/')
def index():
    """Serve the main HTML page"""
    return send_file('index.html')

@app.route('/api/products')
def get_products():
    """Return list of all products"""
    products = ['All Products'] + list(PRODUCT_PROFILES.keys())
    return jsonify(products)

@app.route('/api/data/<product_name>')
def get_data(product_name):
    """Return all dashboard data for a product"""
    if product_name == 'All Products':
        filtered_df = df_reviews
    else:
        filtered_df = df_reviews[df_reviews['product'] == product_name]
    
    total_reviews = len(filtered_df)
    positive_count = len(filtered_df[filtered_df['sentiment_category'] == 'Positive'])
    negative_count = len(filtered_df[filtered_df['sentiment_category'] == 'Negative'])
    positive_pct = (positive_count / total_reviews * 100) if total_reviews > 0 else 0
    
    sentiment_counts = filtered_df['sentiment_category'].value_counts().to_dict()
    pie_data = {
        'labels': list(sentiment_counts.keys()),
        'values': list(sentiment_counts.values())
    }
    
    bar_data = df_reviews.groupby('product')['sentiment_score'].mean().to_dict()
    bar_chart_data = {
        'products': list(bar_data.keys()),
        'scores': list(bar_data.values())
    }
    
    map_data = filtered_df.groupby('country').agg({
        'sentiment_score': 'mean',
        'joy': 'sum',
        'anger': 'sum',
        'sadness': 'sum',
        'fear': 'sum',
        'surprise': 'sum',
        'trust': 'sum',
        'anticipation': 'sum',
        'disgust': 'sum'
    }).reset_index().to_dict('records')
    
    sample_data = filtered_df[['product', 'country', 'review', 'sentiment_score', 'sentiment_category']].head(100).to_dict('records')
    
    return jsonify({
        'metrics': {
            'total_reviews': total_reviews,
            'positive_pct': round(positive_pct, 1),
            'negative_count': negative_count
        },
        'pieChartData': pie_data,
        'barChartData': bar_chart_data,
        'mapChartData': map_data,
        'rawDataSample': sample_data
    })

@app.route('/api/recommendation')
def get_recommendation():
    """Get product recommendation based on country and product"""
    product = request.args.get('product')
    country = request.args.get('country')
    
    if not product or not country:
        return jsonify({'error': 'Missing parameters'}), 400
    
    filtered_df = df_reviews[(df_reviews['product'] == product) & (df_reviews['country'] == country)]
    
    if len(filtered_df) == 0:
        return jsonify({
            'analysis': {
                'top_positive': [],
                'top_negative': []
            },
            'recommendation': f"Insufficient data for {product} in {country}"
        })
    
    all_features = []
    sentiments = []
    for _, row in filtered_df.iterrows():
        for feature in row['features_mentioned']:
            all_features.append(feature)
            sentiments.append(row['sentiment_score'])
    
    feature_sentiment = {}
    for feature, sentiment in zip(all_features, sentiments):
        if feature not in feature_sentiment:
            feature_sentiment[feature] = []
        feature_sentiment[feature].append(sentiment)
    
    avg_feature_sentiment = {k: np.mean(v) for k, v in feature_sentiment.items()}
    
    sorted_features = sorted(avg_feature_sentiment.items(), key=lambda x: x[1], reverse=True)
    
    top_positive = [f[0] for f in sorted_features[:3] if f[1] > 0]
    top_negative = [f[0] for f in sorted_features[-3:] if f[1] < 0]
    top_negative.reverse()
    
    recommendations = PRODUCT_RECOMMENDATIONS.get(product, ['No recommendation available'])
    recommended_product = recommendations[0] if recommendations else 'No recommendation available'
    
    return jsonify({
        'analysis': {
            'top_positive': top_positive,
            'top_negative': top_negative
        },
        'recommendation': recommended_product
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
