# ML Chatbot Streamlit App

This repository contains a Streamlit multi-agent chatbot that answers questions using documents stored in Pinecone.

## Files included

- `app.py` - Streamlit entrypoint
- `head_agent.py` - orchestration logic
- `answering_agent.py`
- `context_rewriter_agent.py`
- `hybrid_filter_agent.py`
- `obnoxious_agent.py`
- `query_agent.py`
- `relevant_docs.py`
- `smalltalk_agent.py`
- `requirements.txt` - Python dependencies for Streamlit Community Cloud
- `runtime.txt` - Python version for deployment
- `.streamlit/config.toml` - Streamlit config
- `secrets_template.toml` - copy these keys into Streamlit Cloud secrets

## Local run

Create a virtual environment, install dependencies, add your secrets locally, then run:

```bash
streamlit run app.py
```

## Required secrets

Add these in Streamlit Cloud app settings:

- `OPENAI_API_KEY`
- `PINECONE_API_KEY`
- `PINECONE_INDEX`
- `PINECONE_NAMESPACE`

## Deploy to Streamlit Community Cloud

1. Push this repo to GitHub.
2. Go to Streamlit Community Cloud.
3. Create a new app from the GitHub repo.
4. Set the main file path to `app.py`.
5. Paste the secrets from `secrets_template.toml` into the app's Secrets section.
6. Deploy.
