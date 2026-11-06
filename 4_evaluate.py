"""Lab 5: evaluate groundedness and relevance (scores 1 to 5, pass threshold 3)."""
import json
import os

from azure.ai.evaluation import GroundednessEvaluator, RelevanceEvaluator

from common import CHAT_DEPLOYMENT, FOUNDRY_API_KEY, OPENAI_ENDPOINT, format_sources
from importlib import import_module

answer = import_module("2_rag_app").answer

# Write your own questions: some answered by your PDFs, one that is NOT in them.
QUESTIONS = [
    "Which role does a developer need to call a model with Microsoft Entra ID?",
    "What does the disableLocalAuth property do?",
    "Which private DNS zones does a Foundry private endpoint use?",
    "What is the minimum subnet size for agent virtual network injection?",
    "Who won the football world cup in 2018?",  # not in the sources
]

model_config = {
    "azure_endpoint": OPENAI_ENDPOINT,
    "azure_deployment": CHAT_DEPLOYMENT,
    "api_version": os.environ.get("EVAL_API_VERSION", "2024-10-21"),
}
kwargs = {}
if FOUNDRY_API_KEY:
    model_config["api_key"] = FOUNDRY_API_KEY
else:
    from azure.identity import DefaultAzureCredential

    kwargs["credential"] = DefaultAzureCredential()

groundedness = GroundednessEvaluator(model_config=model_config, **kwargs)
relevance = RelevanceEvaluator(model_config=model_config, **kwargs)


def score(result: dict, name: str):
    return result.get(name, result.get(f"gpt_{name}"))


rows = []
for q in QUESTIONS:
    text, chunks, _ = answer(q)
    context = format_sources(chunks)
    g = groundedness(query=q, response=text, context=context)
    r = relevance(query=q, response=text)
    rows.append({"query": q, "response": text, "context": context, "groundedness": g, "relevance": r})
    print(f"\nQ: {q}\nA: {text[:200]}")
    print(f"groundedness={score(g, 'groundedness')} relevance={score(r, 'relevance')}")
    print(f"groundedness detail: {g}")

with open("eval_results.jsonl", "w", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps({"query": row["query"], "response": row["response"], "context": row["context"]}) + "\n")
print("\nSaved eval_results.jsonl (query, response, context) for a portal evaluation run.")
