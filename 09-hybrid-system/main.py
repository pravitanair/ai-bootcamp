import os
from dotenv import load_dotenv
from openai import OpenAI

from tools.icd import lookup_icd_code
from tools.reimbursement import estimate_reimbursement
from rag.engine import build_query_engine
from agent.router import classify_query

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

query_engine = build_query_engine()

def synthesize_answer(query: str, evidence: str) -> str:
    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {
                "role": "system",
                "content": (
                    "You are a healthcare AI assistant. "
                    "Use the provided evidence to answer clearly and concisely. "
                    "If evidence is incomplete, say so."
                )
            },
            {
                "role": "user",
                "content": f"Question: {query}\n\nEvidence:\n{evidence}"
            }
        ]
    )
    return response.output_text

query = input("Ask: ")
route = classify_query(query)

print(f"\nRoute selected: {route}")

if route == "icd_lookup":
    term = query.lower().replace("what is the icd code for", "").strip(" ?")
    result = lookup_icd_code(term)
    evidence = str(result)
    answer = synthesize_answer(query, evidence)

elif route == "reimbursement_by_icd":
    words = query.replace("?", "").split()
    icd_code = None
    for w in words:
        if "." in w or w.upper().startswith(("R", "I", "J")):
            icd_code = w
            break

    if icd_code:
        result = estimate_reimbursement(icd_code)
    else:
        result = {"found": False, "message": "No ICD code identified in query."}

    evidence = str(result)
    answer = synthesize_answer(query, evidence)

elif route == "reimbursement_by_symptom":
    term = query.lower().replace("what is the reimbursement for", "").strip(" ?")
    icd_result = lookup_icd_code(term)

    if icd_result.get("found"):
        reimb_result = estimate_reimbursement(icd_result["code"])
        evidence = f"ICD lookup result: {icd_result}\nReimbursement result: {reimb_result}"
    else:
        evidence = f"ICD lookup result: {icd_result}"

    answer = synthesize_answer(query, evidence)

else:
    rag_response = query_engine.query(query)
    evidence = str(rag_response)
    answer = synthesize_answer(query, evidence)

print("\nAnswer:\n")
print(answer)