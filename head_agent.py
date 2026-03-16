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

    def generate_smalltalk_cache(self):
        prompt = """
        Generate a JSON dictionary of more than 500 common greetings (e.g. hello, hi, hey, good morning, etc.) and
        small-talk phrases (e.g. how are you, what's up, who are you, thank you, bye, etc.) mapped to short, friendly responses.
        Use local and colloqual terms (e.g howdy) as appropriate.
        Create it as a single flat list with the greeting/small talk as the key and the response as value.
        Keep responses short and friendly.
        Return ONLY valid JSON.
        """

        resp = self.client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[{"role": "user", "content": prompt}],
        )

        try:
            smalltalk_dict = json.loads(resp.choices[0].message.content)
        except Exception:
            cleaned = resp.choices[0].message.content.strip()
            cleaned = cleaned[cleaned.find("{") : cleaned.rfind("}") + 1]
            smalltalk_dict = json.loads(cleaned)

        return {k.lower(): v for k, v in smalltalk_dict.items()}

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
        text_clean = (text or "").lower().strip().rstrip("!?.,:;")
        return self.small_talk.get(text_clean)

    def main_loop(self, user_query, conversation_history, return_debug: bool = False):
        debug = {"user_query": user_query}

        cached = self.check_smalltalk(user_query)
        debug["smalltalk_tagged"] = bool(cached)
        debug["smalltalk_response_found"] = bool(cached)
        if cached:
            debug["final_answer"] = cached
            return (cached, debug) if return_debug else cached

        llm_smalltalk = self.smalltalk_agent.run(user_query)
        if llm_smalltalk:
            debug["smalltalk_tagged"] = True
            debug["smalltalk_response_found"] = True
            debug["smalltalk_source"] = "llm_agent"
            debug["final_answer"] = llm_smalltalk
            return (llm_smalltalk, debug) if return_debug else llm_smalltalk

        is_obnoxious = self.obnoxious_agent.check_query(user_query)
        debug["obnoxious_tagged"] = bool(is_obnoxious)
        if is_obnoxious:
            resp = "Your message contains inappropriate or obnoxious content."
            debug["final_answer"] = resp
            return (resp, debug) if return_debug else resp

        rewritten_query = self.context_agent.rephrase(
            user_history=conversation_history,
            latest_query=user_query,
        )
        debug["rewritten_query"] = rewritten_query

        hyb = self.hybrid_agent.extract_relevant_subquery(rewritten_query)
        if hyb.get("relevant_query", "").strip() == "":
            resp = "This query is not relevant to the document"
            debug["final_answer"] = resp
            return (resp, debug) if return_debug else resp

        filtered_query = hyb["relevant_query"].strip()
        debug["filtered_query"] = filtered_query

        pinecone_docs = self.query_agent.run(filtered_query)
        debug["retrieved_docs_count"] = len(pinecone_docs) if pinecone_docs else 0

        # Deployment/debug-safe behavior:
        # use retrieved docs directly instead of letting the relevance gate throw out good matches.
        docs_to_use = pinecone_docs
        debug["relevance_any_relevant"] = bool(docs_to_use)
        debug["relevant_docs_count"] = len(docs_to_use) if docs_to_use else 0

        if not docs_to_use:
            resp = "This query is not relevant to the document"
            debug["final_answer"] = resp
            return (resp, debug) if return_debug else resp

        final_answer = self.answer_agent.generate_response(
            query=filtered_query,
            docs=docs_to_use,
            conv_history=conversation_history,
        )
        debug["final_answer"] = final_answer

        return (final_answer, debug) if return_debug else final_answer
