# api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any, Optional
import os

# Import your agent components
try:
    from supervisor import workflow, SupervisorState
    from langchain_core.messages import HumanMessage
    print("✅ Agent components imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    # Create mock classes for testing
    class SupervisorState:
        def __init__(self, **kwargs):
            pass
    workflow = None

app = FastAPI(
    title="Hair Analysis API",
    description="AI-powered hair analysis and recommendation system",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class HairRequest(BaseModel):
    questionnaire: Optional[Dict[str, Any]] = None
    image_path: Optional[str] = None
    user_text: str = "analyze my hair and recommend products"

class HairResponse(BaseModel):
    final_answer: str
    success: bool
    error: Optional[str] = None

@app.get("/")
def read_root():
    return {"message": "Hair Analysis API is running!", "status": "healthy"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "hair-analysis-api"}

@app.post("/analyze", response_model=HairResponse)
def analyze_hair(request: HairRequest):
    try:
        print("Received analysis request")
        
        # For testing without full agent setup
        if workflow is None:
            return HairResponse(
                final_answer="✅ API is working! Agent setup in progress...",
                success=True
            )
        
        # Initialize workflow state
        state = SupervisorState(
            messages=[HumanMessage(content=request.user_text)],
            questionnaire_answers=request.questionnaire or {},
            image_path=request.image_path,
            agent_call_count=1
        )
        
        # Run the agent workflow
        result = workflow.invoke(state)
        final_answer = result.get("final_answer", "No response generated")
        
        return HairResponse(
            final_answer=final_answer,
            success=True
        )
        
    except Exception as e:
        return HairResponse(
            final_answer=f"Error: {str(e)}",
            success=False,
            error=str(e)
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)