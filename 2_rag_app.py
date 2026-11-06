"""Lab 3: retrieve top 3 chunks, answer with the chat model, cite the sources."""
import sys

from common import CHAT_DEPLOYMENT, chat_client, format_sources, retrieve

SYSTEM = (
    "You are an assistant for Microsoft Foundry documentation. "
    "Answer ONLY from the sources below. "
    "After each fact, cite its source number in square brackets, for example [1]. "
    "If the sources do not contain the answer, say: 'I don't know based on the sources.'"
)


def answer(question: str, mode: str = "hybrid", temperature: float = 0.2, top_p: float = 1.0):
    chunks = retrieve(question, mode=mode, k=3)
    sources = format_sources(chunks)
    response = chat_client().responses.create(
        model=CHAT_DEPLOYMENT,
        instructions=SYSTEM,
        input=f"Sources:\n{sources}\n\nQuestion: {question}",
        temperature=temperature,
        top_p=top_p,
    )
    return response.output_text, chunks, response.usage


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "Which role does a developer need to call a model with Microsoft Entra ID?"
    text, chunks, usage = answer(q)
    print(f"Question: {q}\n\nAnswer:\n{text}\n\nSources:")
    for i, c in enumerate(chunks, start=1):
        print(f"[{i}] {c['title']}")
    print(f"\nTokens: input={usage.input_tokens} output={usage.output_tokens}")
