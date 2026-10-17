# src/rag_engine.py

class RAGEngine:
    def __init__(self):
        # මෙහි LangChain අවශ්‍ය නැත. 
        # අවශ්‍ය නම්, පසුව OpenAI API Key එකතු කළ හැක.
        pass

    def generate_explanation(self, cbi_scores, prediction):
        if prediction > 0.5:
            return "The signature matches the reference specimen perfectly."
        else:
            # LLM නොමැතිව සරල පැහැදිලි කිරීමක්
            return f"The signature was flagged as forged due to broken continuity features: {cbi_scores}."