from fastapi import FastAPI, HTTPException
from fastmcp import FastMCP
import httpx
import json
import logging
from tools import calculate_credit_score, calculate_risk

app = FastAPI()

mcp = FastMCP("CreditTools")

@mcp.tool()
async def calculate_credit_score_tool(age: int, income: float, education: str, married: str) -> float:
    """Calculates credit score from client parameters."""
    return calculate_credit_score(age, income, education, married)

@mcp.tool()
async def calculate_risk_tool(metadata: dict) -> float:
    """Estimates risk based on metadata dictionary."""
    return calculate_risk(metadata)

app.mount("/mcp", mcp.http_app())

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OLLAMA_URL = "http://ollama:11434/api/generate"

SYSTEM_PROMPT = """
You are a credit approval assistant. You must analyze the client's characteristics and decide whether to approve a loan.
You have access to two tools:
1. calculate_credit_score – calculates a credit score (0-100) based on parameters.
2. calculate_risk – estimates risk (0-1) from metadata.

Respond ONLY with a JSON object:
{
    "reasoning": "string with your analysis",
    "verdict": 0 or 1 (0 = deny, 1 = approve)
}
"""

@app.post("/approve")
async def approve_loan(client_data: dict):
    """
    Принимает JSON с характеристиками клиента (age, income, education, married и т.п.).
    Вызывает инструменты напрямую, вставляет результаты в промпт и отправляет в LLM.
    """
    age = client_data.get('age', 30)
    income = client_data.get('income', 50000)
    education = client_data.get('education', 'bachelor')
    married = client_data.get('married', 'yes')
    metadata = {
        'education': education,
        'married': married,
        'age': age,
        'income': income
    }
    
    credit_score = calculate_credit_score(age, income, education, married)
    risk = calculate_risk(metadata)
    
    prompt = f"{SYSTEM_PROMPT}\n\nClient data: {json.dumps(client_data)}\n" \
             f"Calculated credit score: {credit_score}\n" \
             f"Calculated risk: {risk}\nDecision:"
    
    async with httpx.AsyncClient() as client:
        payload = {
            "model": "qwen2.5:0.5b",
            "prompt": prompt,
            "stream": False,
            "format": "json", 
            "options": {
                "temperature": 0.2
            }
        }
        logger.info(f"Sending request to Ollama: {payload}")
        try:
            resp = await client.post(OLLAMA_URL, json=payload, timeout=120)
            resp.raise_for_status()
            result = resp.json()
            logger.info(f"Ollama response: {result}")
            llm_output = json.loads(result['response'])
            return llm_output
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"LLM call failed: {e}")

@app.get("/score")
async def get_score(age: int, income: float, education: str, married: str):
    score = calculate_credit_score(age, income, education, married)
    return {"credit_score": score}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)