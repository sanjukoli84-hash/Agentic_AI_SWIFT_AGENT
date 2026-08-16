import re
import json
from typing import Dict, Any
from langchain_ollama import ChatOllama

EXTRACTOR_PROMPT = """You are an expert SWIFT MT103 Message parser.
Your task is to analyze the raw SWIFT message text and extract the fields into a structured JSON object.

Extract the following SWIFT tags if present:
- Tag :20: (Transaction Reference Number) -> "transaction_reference"
- Tag :23B: (Bank Operation Code) -> "bank_operation_code"
- Tag :32A: (Value Date, Currency, and Amount) -> "value_date_currency_amount"
  If possible, split Tag :32A: into:
    - "value_date" (YYMMDD format, e.g., 260815)
    - "currency" (3-letter ISO code, e.g., USD)
    - "amount" (decimal format, replacing comma with dot if necessary, e.g., 50000.00)
- Tag :50K: or :50A: or :50F: (Ordering Customer) -> "ordering_customer"
  Split into:
    - "account" (e.g. /1234567890)
    - "details" (Name and address lines)
- Tag :59: or :59A: (Beneficiary Customer) -> "beneficiary_customer"
  Split into:
    - "account" (e.g. /9876543210 or IBAN)
    - "details" (Name and address lines)
- Tag :71A: (Details of Charges) -> "details_of_charges"

Raw SWIFT message:
\"\"\"
{raw_text}
\"\"\"

Format your output ONLY as a JSON code block. Do not write any conversational text or prefaces.
Example output format:
```json
{{
  "transaction_reference": "TXN12345",
  "bank_operation_code": "CRED",
  "value_date_currency_amount": "260815USD50000,00",
  "value_date": "260815",
  "currency": "USD",
  "amount": "50000.00",
  "ordering_customer": {{
    "account": "/1234567890",
    "details": "JOHN DOE\n123 MAIN STREET\nBRUSSELS, BE"
  }},
  "beneficiary_customer": {{
    "account": "/9876543210",
    "details": "JANE SMITH\n456 OAK AVENUE\nFRANKFURT, DE"
  }},
  "details_of_charges": "SHA",
  "other_fields": {{}}
}}
```
"""

REASONER_PROMPT = """You are a financial transaction compliance officer and SWIFT messaging expert.
Your job is to analyze the parsed SWIFT MT103 fields and the raw text, verify if the SWIFT message is valid or would fail, and compile a compliance auditing report.

Here are the standard verification rules for SWIFT MT103 (Single Customer Credit Transfer):
1. Tag :20: (Transaction Reference) is MANDATORY. It must exist and not be empty.
2. Tag :23B: (Bank Operation Code) is MANDATORY. Typically it must be "CRED" for standard transfers.
3. Tag :32A: (Value Date, Currency, Amount) is MANDATORY.
   - Must contain a valid date in YYMMDD format.
   - Must contain a valid 3-letter ISO Currency.
   - Must contain a numeric Amount.
4. Tag :50K/A/F: (Ordering Customer / Sender) is MANDATORY.
   - Must include an account/identifier (often prefixed with '/').
   - Must include a Name and Address. A missing address or name will cause KYC/AML failures or transaction rejects.
5. Tag :59/A: (Beneficiary Customer / Receiver) is MANDATORY.
   - Must include a Beneficiary Account (IBAN preferred for European countries, or standard account).
   - Must include Beneficiary Name and Address. Missing name or address will cause payment routing failure or compliance rejection.
6. Tag :71A: (Details of Charges) is MANDATORY. Must be "OUR" (sender pays), "BEN" (beneficiary pays), or "SHA" (shared).

Analyze the following parsed data and identify if the message is valid or has omissions:
Parsed Data:
{parsed_data_json}

Raw SWIFT message:
\"\"\"
{raw_text}
\"\"\"

Write a detailed transaction auditing report in Markdown format. Focus on answering:
1. **Status**: Explicitly start with either **PASS (Valid SWIFT Message)** or **FAIL (Invalid SWIFT Message)**.
2. **Missing/Malformed Tags**:
   - If Status is FAIL: List each tag that is missing, empty, or incorrectly formatted. Highlight which fields within the tags are absent (e.g. Beneficiary Account, Ordering Customer Address).
   - If Status is PASS: Explicitly state that all mandatory tags are present and conform to rules.
3. **Operational Consequence**:
   - If Status is FAIL: Explain *why* these missing fields cause the transaction to fail (e.g., "Missing tag :59: means there is no beneficiary designated, which prevents the receiving bank from routing the funds", "Missing ordering customer address violates KYC/AML rules").
   - If Status is PASS: Confirm that the transaction is compliant and ready for routing.
4. **Remediation Steps**:
   - If Status is FAIL: Provide clear instructions on what needs to be added to fix the SWIFT message so it passes successfully.
   - If Status is PASS: State that no further action is required.

Keep the report professional, clear, and actionable. Do not add general pleasantries. Start directly with the report.
"""

def get_llm(model_name: str = "llama3.2", base_url: str = "http://localhost:11434") -> ChatOllama:
    """Instantiate a local ChatOllama LLM."""
    return ChatOllama(
        model=model_name,
        temperature=0.0,
        base_url=base_url
    )

def sanitize_json_string(s: str) -> str:
    """Escapes literal newlines and tabs inside double-quoted string values in a JSON string."""
    result = []
    in_string = False
    escape = False
    for char in s:
        if char == '"' and not escape:
            in_string = not in_string
            result.append(char)
        elif char == '\\' and in_string:
            escape = not escape
            result.append(char)
        elif char == '\n' and in_string:
            result.append('\\n')
            escape = False
        elif char == '\t' and in_string:
            result.append('\\t')
            escape = False
        else:
            escape = False
            result.append(char)
    return "".join(result)

def map_to_tag_keys(extracted: Dict[str, Any]) -> Dict[str, Any]:
    """Ensures both semantic keys and raw SWIFT tag keys (e.g. :20:) are present in the dictionary."""
    mapped = {}
    
    # Map tag :20:
    ref = extracted.get(":20:") or extracted.get("transaction_reference")
    mapped[":20:"] = ref
    mapped["transaction_reference"] = ref
    
    # Map tag :23B:
    op = extracted.get(":23B:") or extracted.get("bank_operation_code")
    mapped[":23B:"] = op
    mapped["bank_operation_code"] = op
    
    # Map tag :32A:
    val_data = extracted.get(":32A:") or extracted.get("value_date_currency_amount")
    if isinstance(val_data, dict):
        mapped[":32A:"] = val_data
        mapped["value_date"] = val_data.get("value_date")
        mapped["currency"] = val_data.get("currency")
        mapped["amount"] = val_data.get("amount")
    else:
        vdate = extracted.get("value_date")
        curr = extracted.get("currency")
        amt = extracted.get("amount")
        # Keep it as a dictionary for tag :32A:
        mapped[":32A:"] = {
            "value_date": vdate,
            "currency": curr,
            "amount": amt,
            "raw": val_data
        }
        mapped["value_date"] = vdate
        mapped["currency"] = curr
        mapped["amount"] = amt
        
    # Map tag :50K:
    order = extracted.get(":50K:") or extracted.get("ordering_customer") or extracted.get(":50A:") or extracted.get(":50F:")
    mapped[":50K:"] = order
    mapped["ordering_customer"] = order
    
    # Map tag :59:
    benef = extracted.get(":59:") or extracted.get("beneficiary_customer") or extracted.get(":59A:")
    mapped[":59:"] = benef
    mapped["beneficiary_customer"] = benef
    
    # Map tag :71A:
    charges = extracted.get(":71A:") or extracted.get("details_of_charges")
    mapped[":71A:"] = charges
    mapped["details_of_charges"] = charges
    
    # Preserve other fields
    for k, v in extracted.items():
        if k not in mapped:
            mapped[k] = v
            
    return mapped

def extract_swift_fields(raw_text: str, llm: ChatOllama) -> Dict[str, Any]:
    """Uses LLM to extract fields from raw SWIFT message, with robust JSON parsing fallbacks."""
    import ast
    prompt = EXTRACTOR_PROMPT.format(raw_text=raw_text)
    
    try:
        response = llm.invoke(prompt)
        response_text = response.content.strip()
        print("\n=== DEBUG: RAW LLM RESPONSE ===")
        print(response_text)
        print("===============================\n")
    except Exception as e:
        return {
            "error": f"Failed to communicate with local Ollama LLM: {str(e)}. Please ensure Ollama is running and the model is pulled."
        }
        
    # Extract text from first curly brace to last curly brace (eliminates conversational prefaces/postscripts)
    start_idx = response_text.find('{')
    end_idx = response_text.rfind('}')
    
    if start_idx != -1 and end_idx != -1:
        json_str = response_text[start_idx:end_idx+1]
    else:
        json_str = response_text

    # Escape literal control characters (like newlines) inside JSON string values
    json_str_sanitized = sanitize_json_string(json_str)

    # Clean up trailing commas which are common syntax errors in LLM outputs
    json_str_cleaned = re.sub(r',\s*\}', '}', json_str_sanitized)
    json_str_cleaned = re.sub(r',\s*\]', ']', json_str_cleaned)

    # Attempt standard JSON loading
    try:
        parsed_json = json.loads(json_str_cleaned)
        return map_to_tag_keys(parsed_json)
    except json.JSONDecodeError:
        # Attempt Python abstract syntax tree parsing as fallback (handles single quotes and Python syntax)
        try:
            parsed_json = ast.literal_eval(json_str_cleaned)
            if isinstance(parsed_json, dict):
                return map_to_tag_keys(parsed_json)
        except Exception:
            pass

    # If both parsers fail, fallback to raw regex extraction
    fallback_json = extract_fields_fallback(raw_text, response_text)
    return map_to_tag_keys(fallback_json)

def extract_fields_fallback(raw_text: str, raw_llm_response: str) -> Dict[str, Any]:
    """Fallback extraction using regular expressions directly on SWIFT text if LLM JSON fails."""
    fields = {}
    
    # Tag :20:
    ref_match = re.search(r':20:([^\s:]+)', raw_text)
    fields["transaction_reference"] = ref_match.group(1) if ref_match else None
    
    # Tag :23B:
    op_match = re.search(r':23B:([^\s:]+)', raw_text)
    fields["bank_operation_code"] = op_match.group(1) if op_match else None
    
    # Tag :32A:
    val_match = re.search(r':32A:([^\n]+)', raw_text)
    if val_match:
        val_str = val_match.group(1).strip()
        fields["value_date_currency_amount"] = val_str
        # Try split e.g. 260815USD50000,00
        parts_match = re.match(r'^(\d{6})([A-Z]{3})([\d,]+)$', val_str)
        if parts_match:
            fields["value_date"] = parts_match.group(1)
            fields["currency"] = parts_match.group(2)
            fields["amount"] = parts_match.group(3).replace(',', '.')
    
    # Tag :50K: (Ordering Customer)
    # Grab until the next tag starting with :
    order_match = re.search(r':50[KAF]:(.*?)(?=\n:[0-9]+[A-Z]?:|$-)', raw_text, re.DOTALL)
    if order_match:
        lines = order_match.group(1).strip().split('\n')
        fields["ordering_customer"] = {
            "account": lines[0] if lines[0].startswith('/') else None,
            "details": "\n".join(lines[1:]) if lines[0].startswith('/') else "\n".join(lines)
        }
        
    # Tag :59: (Beneficiary Customer)
    benef_match = re.search(r':59[A]?:(.*?)(?=\n:[0-9]+[A-Z]?:|$-)', raw_text, re.DOTALL)
    if benef_match:
        lines = benef_match.group(1).strip().split('\n')
        fields["beneficiary_customer"] = {
            "account": lines[0] if lines[0].startswith('/') else None,
            "details": "\n".join(lines[1:]) if lines[0].startswith('/') else "\n".join(lines)
        }
        
    # Tag :71A:
    charges_match = re.search(r':71A:([^\s:]+)', raw_text)
    fields["details_of_charges"] = charges_match.group(1) if charges_match else None
    
    fields["parser_warning"] = "LLM failed to output standard JSON. Applied fallback regex parser."
    fields["raw_llm_response"] = raw_llm_response
    
    return fields

def reason_swift_failures(extracted_fields: Dict[str, Any], raw_text: str, llm: ChatOllama) -> str:
    """Uses LLM to perform logical reasoning on the parsed SWIFT fields to generate a detailed compliance report."""
    if "error" in extracted_fields:
        return f"### Error: Cannot perform reasoning due to preprocessing failure\n\n{extracted_fields['error']}"
        
    parsed_json_str = json.dumps(extracted_fields, indent=2)
    prompt = REASONER_PROMPT.format(
        parsed_data_json=parsed_json_str,
        raw_text=raw_text
    )
    
    # Save debug info locally to inspect what data is being sent to the LLM
    try:
        debug_data = {
            "extracted_fields": extracted_fields,
            "raw_text": raw_text,
            "prompt_sent": prompt
        }
        debug_log_path = os.path.join(os.path.dirname(__file__), "debug_log.json")
        with open(debug_log_path, "w", encoding="utf-8") as df:
            json.dump(debug_data, df, indent=2)
    except Exception as log_err:
        print(f"Failed to write debug log: {log_err}")
    
    try:
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        return f"### Error: Failed to perform reasoning\n\nReason: {str(e)}"
