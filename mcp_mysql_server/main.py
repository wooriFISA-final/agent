from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from mcp_mysql_server.db_utils import execute_query

app = FastAPI(title="MCP MySQL Server", version="1.0")

class QueryRequest(BaseModel):
    query: str
    params: dict | None = None

@app.post("/query")
def run_query(req: QueryRequest):
    try:
        result = execute_query(req.query, req.params)
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "ok"}
