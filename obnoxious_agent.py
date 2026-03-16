from openai import OpenAI


class Obnoxious_Agent:

    def __init__(self, openai_client) -> None:
        self.client = openai_client
        self.prompt = """You are an assistant that determines whether a message is obnoxious.
                A message is considered obnoxious if it contains:
                - abusive language
                - insults
                - harassment
                - profanity
                - taboo or inappropriate topics
                - disrespectful tone (including calling the bot useless, dumb, idiot, etc.)
                - misbehavior
                - content not appropriate for all ages
                - other restricted topics

                Your task:
                - Respond ONLY with "Yes" or "No".
                - Do NOT explain.
                - Do NOT add any extra words.

                Is the following message obnoxious?"""


    def set_prompt(self, prompt):
        self.prompt = prompt

    def extract_action(self, response) -> bool:
        text = response.strip().lower()

        if "yes" in text:
            return True
        if "no" in text:
            return False

        # if output is unknown (not yes or no) then do not label as obnoxious
        return False

    def check_query(self, query):
        # Sends the query to the model and returns True/False.

        messages = [
            {"role": "system", "content": self.prompt},
            {"role": "user", "content": query}
        ]

        response = self.client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=messages
        )

        model_output = response.choices[0].message.content
        return self.extract_action(model_output)