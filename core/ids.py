# core/ids.py
import hashlib

def make_record_id(unit_code: str, element_title: str) -> str:
    s = f"{(unit_code or '').strip()}|{(element_title or '').strip()}"
    return hashlib.sha1(s.encode("utf-8")).hexdigest()
