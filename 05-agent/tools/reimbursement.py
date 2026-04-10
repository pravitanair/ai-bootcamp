import os
import json

def estimate_reimbursement(icd_code:str) -> dict:
    estimates = {
        "R07.9": {"estimated_amount": 125, "currency": "USD"},
        "I10": {"estimated_amount": 95, "currency": "USD"},
        "J45.909": {"estimated_amount": 140, "currency": "USD"},
    }
    key = icd_code.strip().upper()
    if key in estimates:
        return {"found": True, **estimates[key]}
    
    return {"found": False, "message": f"No Reimbursement found for '{icd_code}'"}