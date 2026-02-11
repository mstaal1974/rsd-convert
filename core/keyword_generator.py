from openai import OpenAI
import re

def generate_keywords(client: OpenAI, model: str, skill_statement: str, pcs_text: str):
    prompt = f"""
Generate search-optimised keywords for the following skill.

Skill statement:
{skill_statement}

Performance criteria:
{pcs_text}

Rules:
- Return 8–15 keywords or short phrases
- No full sentences
- Use semicolon-separated format
- Include synonyms where relevant
- Lowercase only
- No duplicates
- No trailing punctuation
Return ONLY the keyword list.
"""

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a skills taxonomy and search optimisation expert."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )

    raw = resp.choices[0].message.content.strip()

    # Clean up formatting
    raw = raw.replace("\n", " ")
    parts = re.split(r";|,", raw)
    cleaned = sorted(set([p.strip().lower() for p in parts if p.strip()]))

    return "; ".join(cleaned)
