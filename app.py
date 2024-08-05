import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from ctransformers import AutoModelForCausalLM
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Đọc file CSV
df = pd.read_csv('dich_vu_chi_ho.csv')

# Khởi tạo TF-IDF Vectorizer
tfidf = TfidfVectorizer(ngram_range=(1, 2))

# Fit TF-IDF trên các câu hỏi trong dataset
tfidf_matrix = tfidf.fit_transform(df['Question'])

# Khởi tạo mô hình ngôn ngữ cho việc tạo sinh câu trả lời
model_path = "vinallama-7b-chat_q5_0.gguf"
try:
    llm = AutoModelForCausalLM.from_pretrained(model_path, model_type="llama")
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    llm = None

def preprocess_query(query):
    return query.lower()

def find_most_similar_question(query, threshold=0.3):
    processed_query = preprocess_query(query)
    query_vector = tfidf.transform([processed_query])
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    best_match_index = cosine_similarities.argmax()
    max_similarity = cosine_similarities[best_match_index]
    if max_similarity > threshold:
        return df.iloc[best_match_index]['Question'], df.iloc[best_match_index]['Answer'], max_similarity
    else:
        return None, None, max_similarity

def generate_answer(question, context):
    if llm is None:
        return "Error: Model not loaded"

    prompt = f"Question: {question}\nContext: {context}\nAnswer:"
    response = llm(prompt, max_new_tokens=512)
    print(f"Generated response: {response}")
    return response

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/ask', methods=['POST'])
def ask_question():
    user_input = request.json.get('question')
    similar_question, direct_answer, similarity = find_most_similar_question(user_input)
    if direct_answer:
        generated_answer = generate_answer(user_input, direct_answer)
        response = {
            'similar_question': similar_question,
            'direct_answer': direct_answer,
            'generated_answer': generated_answer
        }
    else:
        response = {
            'message': 'Không tìm thấy câu hỏi tương tự. Vui lòng thử lại với câu hỏi khác.'
        }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)