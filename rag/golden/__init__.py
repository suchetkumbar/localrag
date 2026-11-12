"""Golden dataset: schema, LLM-assisted drafting, human review, and versioned builds.

Workflow (see data/golden/README.md):

    python -m rag.golden draft      # LLM drafts candidates -> data/golden/drafts.jsonl
    python -m rag.golden review     # a human accepts / edits / rejects each candidate
    python -m rag.golden build      # verified items only -> data/golden/v1.jsonl
    python -m rag.golden validate data/golden/v1.jsonl
"""
