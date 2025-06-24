import google.generativeai as genai

class QAEngine:
    def __init__(self, embedder, vector_store, gemini_api_key):
        self.embedder = embedder
        self.vector_store = vector_store
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel("models/gemini-2.0-flash-lite")

    def answer_question(self, question, top_k=20):
        query_vec = self.embedder.embed([question])[0]
        top_chunks = self.vector_store.search(query_vec, top_k=top_k)

        context = "\n\n".join([chunk for chunk, _ in top_chunks])

        prompt = f"""You are a helpful and intelligent assistant. Use the context provided below to answer the user's question clearly and concisely.

If this is the beginning of the conversation, greet the user warmly. If not, continue the conversation naturally.

If the context does not directly contain the answer, use your broader knowledge base to respond accurately. Do not say you cannot answer the question — try to be as helpful as possible.

Context:
{context}

Question: {question}
Answer:"""

        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"[Error from Gemini] {e}"
