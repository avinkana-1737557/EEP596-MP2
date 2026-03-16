from openai import OpenAI

class Context_Rewriter_Agent:
    def __init__(self, openai_client):

        self.client = openai_client
        self.system_prompt = (
            "You are a context-rewriting assistant. Your job is to rewrite the "
            "user's latest query so that it is fully self-contained and unambiguous. "
            "Resolve pronouns, references, and vague phrases using the conversation "
            "history. Do NOT change the user's intent. Output ONLY the rewritten query."
            "Don't summarize the users message either. Just ensure that the user query "
            "makes sense by adding any necessary past context."
        )

    def rephrase(self, user_history, latest_query):
        """
        Rewrites the latest user query using the conversation history.

        Parameters:
            user_history (list[str]): A list of previous user messages.
            latest_query (str): The most recent user query.

        Returns:
            str: A rewritten, unambiguous version of the query.
        """

        # Combine history into a readable block
        history_text = "\n".join([f"User: {msg}" for msg in user_history])

        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": (
                    f"Conversation history:\n{history_text}\n\n"
                    f"Latest query: {latest_query}\n\n"
                    "Rewrite the latest query so it is fully explicit."
                )
            }
        ]

        response = self.client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=messages
        )

        rewritten = response.choices[0].message.content.strip()
        return rewritten