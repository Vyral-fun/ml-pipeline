from app import api_app
import os
import uvicorn
from dotenv import load_dotenv

# Load environment variables from api_config.env
load_dotenv(dotenv_path="api_config.env")

if __name__ == "__main__":
    uvicorn.run(
        "api_server:api_app",
        host="0.0.0.0",
        port=int(os.getenv("API_PORT", 8002)),
        reload=True
    )
