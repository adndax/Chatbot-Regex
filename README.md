# Chat Conversation Segmentation and Semantic Chatbot

This project demonstrates how chat conversations can be segmented into question-and-answer (Q&A) pairs using regular expressions, and how those segments can be used to build a simple chatbot based on semantic similarity.

## Features

- Regex-based segmentation of real-world human conversation logs
- Data cleaning and preprocessing with Python
- Statistical analysis of Q&A patterns
- SentenceTransformer-powered chatbot for semantic response
- Lightweight, fast, and easy to deploy

## Dataset

- Source: [Human Conversation Training Data](https://www.kaggle.com/datasets/projjal1/human-conversation-training-data)
- Format: Plain text, alternating dialogue between Human 1 and Human 2

## How It Works

1. **Preprocessing:** Reads and cleans the chat dataset into a structured format.
2. **Regex Segmentation:** Uses custom regex to detect questions and associate them with their answers.
3. **Analysis:** Calculates statistical metrics such as number of answers per question.
4. **Chatbot:** A user can ask a question, and the bot will find the most semantically similar past question and return its corresponding answer.

## Requirements

- Python 3.7+
- `pandas`
- `re`
- `ast`
- `sentence-transformers`

Install dependencies using:

```bash
pip install pandas sentence-transformers
```

## Run the Chatbot
```
python chatbot.py
```
Then start typing your questions. Type exit or quit to stop the chatbot.

## License
This project is for educational and research purposes only. Dataset belongs to its respective creator.
