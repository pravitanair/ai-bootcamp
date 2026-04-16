def classify_query(query: str) -> str:
    q = query.lower()

    if "icd" in q and "reimbursement" not in q:
        return "icd_lookup"

    if "reimbursement" in q and "icd" in q:
        return "reimbursement_by_icd"

    if "reimbursement" in q:
        return "reimbursement_by_symptom"

    return "rag"