import os
import json

# returns a dictionary


def lookup_icd_code(term: str) -> dict:
    # This creates a Python dictionary called codes.
    codes = {
        "chest pain": {"code": "R07.9", "description": "Chest pain, unspecified"},
        "hypertension": {"code": "I10", "description": "Essential (primary) hypertension"},
        "asthma": {"code": "J45.909", "description": "Unspecified asthma, uncomplicated"},
    }

    key = term.strip().lower()
    if key in codes:
        # The ** operator is dictionary unpacking.
        # Ex.     return {"found": False, "message": f"No ICD code found for '{term}'"}
        return {"found": True, **codes[key]}
    return {"found": False, "message": f"No ICD code found for '{term}'"}