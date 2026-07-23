from src.hybrid_retriever import hybrid_search
from src.reranker import rerank

from src.prompt_engineering import build_prompt

query = "What is the quantity of Cooling Towers?"

# Chapter 9
retrieved = hybrid_search(query)

# Chapter 10
reranked = rerank(query, retrieved)

# Chapter 11
prompt = build_prompt(query, reranked)

print(prompt)