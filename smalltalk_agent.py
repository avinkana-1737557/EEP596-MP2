import re
from typing import Optional


class SmallTalk_Agent:
    """
    Small talk detection + 1-sentence response generator.
    Designed to be called ONLY if the fast dictionary lookup fails.
    """

    def __init__(self, openai_client) -> None:
        self.client = openai_client

        # Light gate to reduce unnecessary LLM calls
        self.ml_keywords = [
            "gradient", "descent", "regression", "classification", "loss", "model",
            "overfitting", "underfitting", "neural", "network", "transformer",
            "embedding", "token", "attention", "svm", "logistic", "linear",
            "precision", "recall", "f1", "auc", "confusion matrix", "dataset"
        ]
        self.task_verbs = ["explain", "define", "compare", "implement", "write", "code", "calculate", "solve", "derive"]

    def _normalize(self, text: str) -> str:
        t = (text or "").lower().strip()
        t = re.sub(r"[^a-z0-9'\s]+", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
        return t

    def looks_like_smalltalk_candidate(self, user_query: str) -> bool:
        """
        Cheap gate: avoid spending an LLM call on obvious non-smalltalk.
        """
        t = self._normalize(user_query)

        if not t:
            return False

        # If the query is long then it is likely not smalltalk
        if len(t) > 140:
            return False

        # If the query contains any ML keywords then it is likely not smalltalk
        if any(k in t for k in self.ml_keywords):
            return False

        # If the query contains words such as 'explain', 'identify', etc. then it is probably not smalltalk
        for v in self.task_verbs:
            if re.search(rf"\b{re.escape(v)}\b", t):
                return False

        return True

    def is_smalltalk(self, user_query: str) -> bool:
        """
        LLM classifier: returns True if SMALLTALK else False.
        Must output strictly SMALLTALK or NOT_SMALLTALK.
        """
        prompt = f"""
Classify the user message as SMALLTALK or NOT_SMALLTALK.

SMALLTALK includes greetings, pleasantries, simple small talk.
NOT_SMALLTALK includes any request for information, tasks, or questions about machine learning.
NOT_SMALLTALK also includes any questions that are not simple greetings. Anything that requires
looking up information to respond to should be categorized as NOT_SMALLTALK.
Only true smalltalk such as simple greetings (eg. 'Hello', 'Good morning', etc.)
SMALLTALK should also include asking what the model can do ie. 'What can you do?'

User message: "{user_query}"

Respond with ONE word only: SMALLTALK or NOT_SMALLTALK
""".strip()

        resp = self.client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        out = (resp.choices[0].message.content or "").strip().upper()
        return out.startswith("SMALLTALK")

    def generate_one_sentence_reply(self, user_query: str) -> str:
        """
        Generate a natural 1-sentence response for small talk.
        """
        prompt = f"""
You are a friendly assistant. Reply to the user's message naturally.

Rules:
- ONE sentence only.
- Friendly tone.
- Do NOT ask multiple questions.
- Do NOT mention policies or being an AI.
- No bullet points, no quotes, no extra formatting.

User message: "{user_query}"
""".strip()

        resp = self.client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        return (resp.choices[0].message.content or "").strip()

    def run(self, user_query: str) -> Optional[str]:
        """
        Returns a 1-sentence smalltalk response if it is smalltalk.
        Returns None if NOT smalltalk or if it doesn't pass the candidate gate.
        """
        if not self.looks_like_smalltalk_candidate(user_query):
            return None

        if not self.is_smalltalk(user_query):
            return None

        return self.generate_one_sentence_reply(user_query)