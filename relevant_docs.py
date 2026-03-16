from openai import OpenAI


class Relevant_Documents_Agent:
    def __init__(self, openai_client) -> None:

        self.client = openai_client

        self.system_prompt = (
            "You are a relevance-checking assistant. Your job is to determine whether "
            "the retrieved documents are relevant to the user's query. "
            "Respond ONLY with 'Yes' or 'No'."
        )

    def get_relevance(self, conversation) -> str:

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": conversation}
        ]

        response = self.client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=messages
        )

        return response.choices[0].message.content.strip()

    def run(self, query, retrieved_docs):
        """
        Main entry point for the Head Agent.

        Steps:
        1. Format the conversation (query + docs)
        2. Ask GPT if the docs are relevant
        3. If relevant → return docs
        4. If not → return empty list
        """

        # Format documents into readable text
        if not retrieved_docs:
            return []

        formatted_docs = []
        for i, doc in enumerate(retrieved_docs):
            if hasattr(doc, "page_content"):
                formatted_docs.append(f"[Doc {i+1}] {doc.page_content}")
            elif hasattr(doc, "metadata") and "text" in doc.metadata:
                formatted_docs.append(f"[Doc {i+1}] {doc.metadata['text']}")
            else:
                formatted_docs.append(f"[Doc {i+1}] {str(doc)}")

        docs_text = "\n\n".join(formatted_docs)

        print(docs_text)

        conversation = f"""
                      User Query:
                      {query}

                      Retrieved Documents:
                      {docs_text}

                      Are these documents relevant to the query?
                      Respond ONLY with Yes or No.
                      For context, the documents will be pulled from a machine learning textbook.
                      Queries should be clearly related to machine learning or not.
                      If there is any relevance between the retrieved documents and the query, 
                      you should respond with Yes. Else respond with No.
                      """

        relevance = self.get_relevance(conversation)

        if relevance.lower().startswith("yes"):
            return retrieved_docs
        else:
            return []