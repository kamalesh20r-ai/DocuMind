from dotenv import load_dotenv
import os
load_dotenv()
api_key=os.getenv("GEMINI_API_KEY")
if api_key :
    print("Key Loaded Sucessfully")
else :
    print("Key Not Found")