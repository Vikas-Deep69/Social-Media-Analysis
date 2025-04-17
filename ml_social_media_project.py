import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, r2_score
from sklearn.feature_extraction.text import TfidfVectorizer

# Set random seeds
np.random.seed(42)
random.seed(42)

# 1. Influencer Impact Analysis
def generate_better_influencer_data(n=1000):
    followers = np.random.randint(1000, 1000000, n)
    engagement_rate = np.random.uniform(0.01, 0.15, n)
    niche_score = np.random.uniform(0, 1, n)
    content_quality = np.random.uniform(0, 1, n)
    influencer_score = followers * engagement_rate * (0.4 * niche_score + 0.6 * content_quality)
    high_sales = influencer_score > np.percentile(influencer_score, 60)
    return pd.DataFrame({
        'followers': followers,
        'engagement_rate': engagement_rate,
        'niche_score': niche_score,
        'content_quality': content_quality,
        'high_sales': high_sales.astype(int)
    })

df_influencer = generate_better_influencer_data()
X_inf = df_influencer.drop(columns='high_sales')
y_inf = df_influencer['high_sales']
X_inf_train, X_inf_test, y_inf_train, y_inf_test = train_test_split(X_inf, y_inf, test_size=0.2, random_state=42)
model_influencer = GradientBoostingClassifier()
model_influencer.fit(X_inf_train, y_inf_train)
acc_inf = accuracy_score(y_inf_test, model_influencer.predict(X_inf_test))
pickle.dump(model_influencer, open("influencer_model.pkl", "wb"))

# Influencer Visualization - Correlation Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df_influencer.corr(), annot=True, cmap="Blues")
plt.title("Influencer Features Correlation Heatmap")
plt.tight_layout()
plt.savefig("influencer_heatmap.png")
plt.close()

# 2. Content Performance Analytics
def generate_content_data(n=1000):
    length = np.random.randint(20, 200, n)
    engagement = np.random.uniform(0, 1, n)
    novelty = np.random.uniform(0, 1, n)
    shares = (length * 0.5 + engagement * 1000 + novelty * 500 + np.random.normal(0, 100, n)).astype(int)
    return pd.DataFrame({
        'length': length,
        'engagement': engagement,
        'novelty': novelty,
        'shares': shares
    })

df_content = generate_content_data()
X_con = df_content.drop(columns='shares')
y_con = df_content['shares']
X_con_train, X_con_test, y_con_train, y_con_test = train_test_split(X_con, y_con, test_size=0.2, random_state=42)
model_content = RandomForestRegressor()
model_content.fit(X_con_train, y_con_train)
r2_con = r2_score(y_con_test, model_content.predict(X_con_test))
pickle.dump(model_content, open("content_model.pkl", "wb"))

# Content Shares Histogram
plt.figure(figsize=(8, 5))
sns.histplot(df_content['shares'], kde=True, bins=30)
plt.title("Content Shares Distribution")
plt.xlabel("Shares")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("content_histogram.png")
plt.close()

# 3. Sentiment Analysis
def generate_sentiment_data(n=1000):
    texts = ['great product', 'bad experience', 'loved it', 'terrible service', 'excellent quality', 'worst ever', 
             'amazing', 'not good', 'fantastic', 'horrible']
    sentiments = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    data = [random.choice(list(zip(texts, sentiments))) for _ in range(n)]
    return pd.DataFrame(data, columns=['text', 'sentiment'])

df_sentiment = generate_sentiment_data()
X_sent = df_sentiment['text']
y_sent = df_sentiment['sentiment']
X_sent_train, X_sent_test, y_sent_train, y_sent_test = train_test_split(X_sent, y_sent, test_size=0.2, random_state=42)
model_sentiment = make_pipeline(TfidfVectorizer(), GradientBoostingClassifier())
model_sentiment.fit(X_sent_train, y_sent_train)
acc_sent = accuracy_score(y_sent_test, model_sentiment.predict(X_sent_test))
pickle.dump(model_sentiment, open("sentiment_model.pkl", "wb"))

# Sentiment Bar Chart
plt.figure(figsize=(6, 4))
sns.countplot(x='sentiment', data=df_sentiment)
plt.title("Sentiment Label Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("sentiment_bar.png")
plt.close()

# Summary Output
print(f"Influencer Impact Accuracy: {acc_inf * 100:.2f}%")
print(f"Content Performance R2 Score: {r2_con:.2f}")
print(f"Sentiment Analysis Accuracy: {acc_sent * 100:.2f}%")
