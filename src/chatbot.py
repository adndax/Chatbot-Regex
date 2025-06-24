from sentence_transformers import SentenceTransformer, util
import pandas as pd

model = SentenceTransformer('all-MiniLM-L6-v2')

df = pd.read_csv("../data/segmented_chat.csv")
import ast
df['answers'] = df['answers'].apply(ast.literal_eval)

questions = df['question'].str.replace(r'^Human \d+: ', '', regex=True)
question_embeddings = model.encode(questions.tolist(), convert_to_tensor=True)

def chatbot_response(user_input):
    user_embedding = model.encode(user_input, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(user_embedding, question_embeddings)
    
    best_match_idx = cosine_scores.argmax().item()
    return df.loc[best_match_idx, 'answers'][0]

while True:
    user_input = input("You: ")
    if user_input.lower() in ['exit', 'quit']:
        print("Bot: Goodbye!")
        break
    print("Bot:", chatbot_response(user_input))


