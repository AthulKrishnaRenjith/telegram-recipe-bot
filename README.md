# Telegram Recipe Bot Using Machine Learning

> 🥘 Your AI-powered kitchen assistant! Snap a photo or type ingredients—get personalized recipe recommendations instantly.

## 🚀 Overview

This project is a machine learning-based Telegram bot that intelligently recommends recipes based on user-provided ingredients—typed or photographed. Built using a hybrid NLP and computer vision pipeline, the bot aims to promote healthy eating, minimize food waste, and provide a smart shopping assistant.

Developed as part of a Bachelor's capstone project at APJ Abdul Kalam Technological University, this system demonstrates practical integration of deep learning, NLP, cloud messaging platforms, and containerized deployment.

## 👩‍💼 Author & Team
- **Athul Krishna** (Project Lead, ML Engineering)
- Ashling Vahab
- Fresnel Fabian
- Reenumol S

> Supervised by Asst. Prof. Manoj M J and Asst. Prof. Rakhi Roy J

## 🎯 Features
- **Image Recognition**: Classify ingredients using ResNet-based CNN model
- **NLP Engine**: Extract attributes from recipes using Doc2Vec & Word2Vec
- **Recipe Recommendation**: Suggests similar and creative recipes from available ingredients
- **Cuisine Prediction**: Logistic regression on TF-IDF embedded data for cuisine classification
- **Interactive Telegram Bot**: Built using Telegram Bot API with inline button handling
- **Dockerized Setup**: Fast deployment on any compatible machine

## 📝 Technical Stack
- **Languages**: Python, Bash
- **Libraries**: Scikit-learn, Gensim, TensorFlow/Keras, NLTK, OpenCV, Pillow
- **Frameworks**: Telegram Bot API, Docker
- **NLP Models**: Doc2Vec, TF-IDF, Logistic Regression
- **CV Models**: ResNet-50
- **Deployment**: Docker + Telegram API

## 🎡 Use Cases
- Real-time cooking assistant for households
- Dietary guidance based on food classification
- Grocery planning assistant
- Personalized recipe discovery

## ⚖️ ML Models and Methodologies
- **Image Classifier**: Fine-tuned ResNet50 for ingredient detection
- **Text Classifier**: Cuisine prediction via multinomial logistic regression
- **Similarity Engine**: Doc2Vec for matching ingredient profiles
- **Data Preprocessing**: Tokenization, Lemmatization, Stopword removal

## 🤖 Getting Started
### With Docker
```bash
docker pull tylerdurden1291/telegram-bot-recipes:mainimage
docker run -e token=<your_token> tylerdurden1291/telegram-bot-recipes:mainimage
```

### Without Docker
```bash
pip install -r requirements.txt
bash setup.sh
python train_and_populate_recommendation_db.py
python app.py <your_token>
```

## 🌐 Dataset
- Collected over **1.8 GB of food images and text data** using **Common Crawl** and **Scrapy**.
- Custom dataset created for subcontinental dishes

## 🎓 Academic Context
Submitted in partial fulfillment for the **Bachelor of Technology in Computer Science and Engineering** under **APJ Abdul Kalam Technological University**, August 2022.

## 🔗 References
- [Telegram Bot API](https://core.telegram.org/bots/api)
- [ResNet Paper](https://arxiv.org/abs/1512.03385)
- [Doc2Vec Gensim](https://radimrehurek.com/gensim/models/doc2vec.html)
- [Scikit-learn Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

---

> For collaborations, feel free to reach out via [LinkedIn](https://www.linkedin.com/in/athulkrishnarenjith/) or email.

---
