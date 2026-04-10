import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


SYSTEM_INSTRUCTIONS = """
You are a healthcare AI assistant.

Rules:
- Use tools when needed.
- If the user asks for an ICD code for a symptom, use lookup_icd_code.
- If the user asks for reimbursement for an ICD code, use estimate_reimbursement.
- If the user asks for reimbursement for a symptom, first find the ICD code, then estimate reimbursement.
- Do not guess missing data.
- If a tool returns no result, say "insufficient data".
- Keep the final answer clear and concise.
"""

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


tools = [
    {
        "type": "function",
        "name": "lookup_icd_code",
        "description": "Look up an ICD code for a diagnosis or symptom term.",
        "parameters":{
            "type": "object",
            "properties": {
                "term": {
                    "type": "string",
                    "description": "Diagnosis or symptom term to look up"
                }
            },
            "required": ["term"],
            "additionalProperties": False
        }
    },
    {
        "type": "function",
        "name": "estimate_reimbursement",
        "description": "Look up reimbursement amount for an ICD code.",
        "parameters":{
            "type": "object",
            "properties": {
                "icd_code": {
                    "type": "string",
                    "description": "ICD code to look up"
                }
            },
            "required": ["icd_code"],
            "additionalProperties": False
        }
    }
]

def run_tool(tool_name: str, args: dict) -> dict:
  
    if tool_name == "lookup_icd_code":
        term = args.get("term", "").strip() 
        if not term:
            return {"found": False, "message": "Missing required field: term"}
        return lookup_icd_code(term)
    if tool_name == "estimate_reimbursement":
        icd_code = args.get("icd_code", "").strip() 
        if not icd_code:
            return {"found": False, "message": "Missing required field: icd_code"}
        return estimate_reimbursement(icd_code)    



user_input = input("Ask: ")

response = client.responses.create(
    model="gpt-4.1-mini",
    instructions=SYSTEM_INSTRUCTIONS,
    input=user_input,
    tools=tools
)

while True:
    function_calls = [item for item in response.output if item.type == "function_call"]

    if not function_calls:
        print("\nFinal answer:\n")
        print(response.output_text)
        break

    tool_outputs = []
    for call in function_calls:
        print("\nModel decided to call tool:")
        print(f"Tool: {call.name}")
        print(f"Arguments: {call.arguments}")
        


        args = json.loads(call.arguments)
        tool_result = run_tool(call.name, args)
        print("Result:", tool_result)

        tool_outputs.append(
            {
                "type": "function_call_output",
                "call_id": call.call_id,
                "output": json.dumps(tool_result)
            }
        )
    response = client.responses.create(
        model="gpt-4.1-mini",
        instructions=SYSTEM_INSTRUCTIONS,
        previous_response_id=response.id,
        input=tool_outputs,
        tools=tools
    )

    print("\nFinal answer:\n")
    print(response.output_text)


