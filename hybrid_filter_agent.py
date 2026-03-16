class Hybrid_Filter_Agent:
    def __init__(self, openai_client) -> None:
        self.client = openai_client

    def extract_relevant_subquery(self, user_query: str) -> dict:
        """
        Returns dict:
          {
            "is_hybrid": bool,
            "relevant_query": str   # empty if none
          }
        """
        prompt = f"""
You are filtering a user message for a document-grounded QA bot.

Goal:
- Keep ONLY the part of the user's message that is relevant to a machine-learning document/Q&A system.
- Remove/ignore any irrelevant parts (sports, celebrities, travel, cooking, personal requests, etc.).
- If the user message contains both relevant AND irrelevant parts, set is_hybrid=true.
- If it contains NO relevant part, set relevant_query="".

Return ONLY valid JSON with keys:
- is_hybrid (true/false)
- relevant_query (string)

User message:
"{user_query}"
""".strip()

        resp = self.client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        text = (resp.choices[0].message.content or "").strip()

        import json, re
        try:
            return json.loads(text)
        except Exception:
            m = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if not m:
                return {"is_hybrid": False, "relevant_query": user_query}
            return json.loads(m.group(0))