from openai import OpenAI
from langchain_openai import OpenAIEmbeddings

import json


class Head_Agent:
    def __init__(self, openai_key, pinecone_key, pinecone_index_name, pinecone_namespace) -> None:
        self.openai_key = openai_key
        self.pinecone_key = pinecone_key
        self.index_name = pinecone_index_name
        self.index_ns = pinecone_namespace

        self.client = None

        self.obnoxious_agent = None
        self.context_agent = None
        self.relevant_agent = None
        self.query_agent = None
        self.answer_agent = None
        self.smalltalk_agent = None
        self.hybrid_agent = None
        self.small_talk = {}

    def _fallback_smalltalk_cache(self):
        return {
            "hi": "Hi! How can I help with machine learning today?",
            "hello": "Hello! How can I help with machine learning today?",
            "hey": "Hey! How can I help with machine learning today?",
            "good morning": "Good morning! How can I help with machine learning today?",
            "good afternoon": "Good afternoon! How can I help with machine learning today?",
            "good evening": "Good evening! How can I help with machine learning today?",
            "how are you": "I'm doing well, thanks! How can I help with machine learning today?",
            "whats up": "Not much—ready to help with machine learning questions.",
            "what's up": "Not much—ready to help with machine learning questions.",
            "who are you": "I'm a machine learning chatbot built to answer questions from the course material.",
            "thank you": "You're welcome!",
            "thanks": "You're welcome!",
            "bye": "Bye!",
            "goodbye": "Goodbye!",
            "see you": "See you later!",
        }

    def generate_smalltalk_cache(self):
        prompt = """
        Generate a JSON dictionary of common greetings and small-talk phrases mapped to short, friendly responses.
        Keep the dictionary under 100 entries.
        Return ONLY valid JSON.
        """

        try:
            resp = self.client.chat.completions.create(
                model="gpt-4.1-nano",
                messages=[{"role": "user", "content": prompt}],
            )
            content = (resp.choices[0].message.content or "").strip()
            if not content:
                return self._fallback_smalltalk_cache()

            try:
                smalltalk_dict = json.loads(content)
            except Exception:
                start = content.find("{")
                end = content.rfind("}")
                if start == -1 or end == -1 or end <= start:
                    return self._fallback_smalltalk_cache()
                smalltalk_dict = json.loads(content[start:end + 1])

            if not isinstance(smalltalk_dict, dict) or not smalltalk_dict:
                return self._fallback_smalltalk_cache()

            normalized = {}
            for k, v in smalltalk_dict.items():
                if isinstance(k, str) and isinstance(v, str):
                    normalized[k.lower().strip()] = v.strip()

            return normalized or self._fallback_smalltalk_cache()
        except Exception:
            return self._fallback_smalltalk_cache()

    def setup_sub_agents(self):
        self.client = OpenAI(api_key=self.openai_key)
        self.small_talk = self.generate_smalltalk_cache()
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=self.openai_key,
        )

        from obnoxious_agent import Obnoxious_Agent
        from context_rewriter_agent import Context_Rewriter_Agent
        from relevant_docs import Relevant_Documents_Agent
        from query_agent import Query_Agent
        from answering_agent import Answering_Agent
        from smalltalk_agent import SmallTalk_Agent
        from hybrid_filter_agent import Hybrid_Filter_Agent

        self.obnoxious_agent = Obnoxious_Agent(self.client)
        self.context_agent = Context_Rewriter_Agent(self.client)
        self.relevant_agent = Relevant_Documents_Agent(self.client)
        self.query_agent = Query_Agent(
            self.pinecone_key,
            self.index_name,
            self.index_ns,
            self.client,
            embeddings,
        )
        self.answer_agent = Answering_Agent(self.client)
        self.smalltalk_agent = SmallTalk_Agent(self.client)
        self.hybrid_agent = Hybrid_Filter_Agent(self.client)

    def check_smalltalk(self, text: str):
        text = text.lower().strip()
        text_clean = text.rstrip("!?.,:;")
        return self.small_talk.get(text_clean)

    def main_loop(self, user_query, conversation_history, return_debug: bool = False):
        debug = {"user_query": user_query}

        cached = self.check_smalltalk(user_query)
        debug["smalltalk_tagged"] = True if cached else False
        debug["smalltalk_response_found"] = True if cached else False
        if cached:
            debug["final_answer"] = cached
            if return_debug:
                return cached, debug
            return cached

        llm_smalltalk = self.smalltalk_agent.run(user_query)
        if llm_smalltalk:
            if return_debug:
                debug["smalltalk_tagged"] = True
                debug["smalltalk_response_found"] = True
                debug["smalltalk_source"] = "llm_agent"
                debug["final_answer"] = llm_smalltalk
                return llm_smalltalk, debug
            return llm_smalltalk

        is_obnoxious = self.obnoxious_agent.check_query(user_query)
        debug["obnoxious_tagged"] = bool(is_obnoxious)
        if is_obnoxious:
            resp = "Your message contains inappropriate or obnoxious content."
            debug["final_answer"] = resp
            if return_debug:
                return resp, debug
            return resp

        rewritten_query = self.context_agent.rephrase(
            user_history=conversation_history,
            latest_query=user_query,
        )
        debug["rewritten_query"] = rewritten_query

        hyb = self.hybrid_agent.extract_relevant_subquery(rewritten_query)
        if hyb.get("relevant_query", "").strip() == "":
            resp = "This query is not relevant to the document"
            debug["final_answer"] = resp
            if return_debug:
                return resp, debug
            return resp

        filtered_query = hyb["relevant_query"].strip()

        pinecone_docs = self.query_agent.run(filtered_query)
        debug["retrieved_docs_count"] = len(pinecone_docs) if pinecone_docs else 0

        docs_to_use = pinecone_docs
        debug["relevance_any_relevant"] = True if docs_to_use else False
        debug["relevant_docs_count"] = len(docs_to_use) if docs_to_use else 0

        if not docs_to_use:
            resp = "This query is not relevant to the document"
            debug["final_answer"] = resp
            if return_debug:
                return resp, debug
            return resp

        final_answer = self.answer_agent.generate_response(
            query=filtered_query,
            docs=docs_to_use,
            conv_history=conversation_history,
        )
        debug["final_answer"] = final_answer

        if return_debug:
            return final_answer, debug
        return final_answer