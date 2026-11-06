"""Lab 4: change one setting at a time and note the effect."""
import json

from common import CHAT_DEPLOYMENT, chat_client, format_sources, retrieve

QUESTION = "Explain the difference between api key and access token authentications."
chunks = retrieve(QUESTION, k=3)
SOURCES = format_sources(chunks)
client = chat_client()

GROUNDED = "Answer only from the sources and cite them as [n]. If the answer is not in the sources, say so."
PIRATE = "Answer like a pirate, in two sentences, without citations."

SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "citations": {"type": "array", "items": {"type": "integer"}},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
    },
    "required": ["answer", "citations", "confidence"],
    "additionalProperties": False,
}


def run(label, system=GROUNDED, temperature=0.2, top_p=1.0, text_format=None, repeat=1):
    print(f"\n===== {label} (temperature={temperature}, top_p={top_p}) =====")
    for n in range(repeat):
        kwargs = dict(
            model=CHAT_DEPLOYMENT,
            instructions=system,
            input=f"Sources:\n{SOURCES}\n\nQuestion: {QUESTION}",
            temperature=temperature,
            top_p=top_p,
        )
        if text_format:
            kwargs["text"] = {"format": text_format}
        out = client.responses.create(**kwargs).output_text
        print(f"--- run {n + 1} ---\n{out[:600]}")
        if text_format:
            data = json.loads(out)  # fails if the output is not valid JSON
            print(f"Parsed keys: {list(data)}")


run("A. Baseline", temperature=0.2, repeat=2)
run("B. High temperature", temperature=1.5, repeat=2)
run("C. Low top_p", temperature=1.0, top_p=0.1, repeat=2)
run("D. Different system message", system=PIRATE)
run(
    "E. JSON schema output",
    text_format={"type": "json_schema", "name": "rag_answer", "schema": SCHEMA, "strict": True},
)
