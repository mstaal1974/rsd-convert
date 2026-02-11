from openai import OpenAI
from .qa import qa_check

def build_bart_prompt(unit_code, unit_title, element_title, pcs_text) -> str:
    return f"""
Transform the following Performance Criteria into ONE element-level skill statement using BART.

Certification: {unit_code} {unit_title}
Element: {element_title}
Performance Criteria:
{pcs_text}

Rules:
- Return ONE sentence (25–45 words).
- Must include a method phrase (by/using/through/via) and an outcome phrase (to ensure/to enable/to support/so that).
- Do NOT copy PCs verbatim; abstract into professional capability language.
Return ONLY the final skill statement.
""".strip()

def build_fix_prompt(unit_code, unit_title, element_title, pcs_text, current_skill) -> str:
    return f"""
Rewrite the skill statement to comply with ALL constraints.

Certification: {unit_code} {unit_title}
Element: {element_title}
Performance Criteria:
{pcs_text}

Current skill statement:
{current_skill}

Constraints:
- ONE sentence, 25–45 words.
- Must include method phrase (by/using/through/via).
- Must include outcome phrase (to ensure/to enable/to support/so that).
- Keep atomic, assessable, professional capability language.
Return ONLY the revised sentence.
""".strip()

def generate_skill_statement(client: OpenAI, model: str, unit_code, unit_title, element_title, pcs_text, max_fixes: int = 2):
    prompt = build_bart_prompt(unit_code, unit_title, element_title, pcs_text)
    resp = client.responses.create(model=model, input=prompt)
    skill = resp.output_text.strip()

    qa = qa_check(skill)
    fixes = 0
    while not qa["passes"] and fixes < max_fixes:
        fix_prompt = build_fix_prompt(unit_code, unit_title, element_title, pcs_text, skill)
        fix = client.responses.create(model=model, input=fix_prompt)
        skill = fix.output_text.strip()
        qa = qa_check(skill)
        fixes += 1

    return skill, qa, prompt
