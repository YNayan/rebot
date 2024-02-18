import os
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import ChatPromptTemplate
from typing import Union

from fastapi import FastAPI, status, Depends, HTTPException, Response


class CodeInput(BaseModel):
    code:str

class CodeOutput(BaseModel):
    refactor_code:str
    language:str

class ErrorResponse(BaseModel):
    status_code: int
    message: str

# loading the api key from env file
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if OPENAI_API_KEY is None:
    raise EnvironmentError("OpenAI API key is missing. Please set the OPENAI_API_KEY environment variable")

# initializing the model 
CHAT_MODEL = "gpt-3.5-turbo"
try: 
    chat_model = ChatOpenAI(temperature=0.0, model=CHAT_MODEL)
except Exception as e:
    raise RuntimeError("Failed to initialize ChatOpenAI instance: {}".format(str(e)))


app = FastAPI()

CODE_TEMPLATE = """Refactor the code based on the following criteria\
                    imporve variable function and class name to be descriptive and meaningful \
                    update the variable name whereever they are used \
                    use consistent formating , indentation and commenting \
                    follow the best practices for the respective programming language \
                    group related code together and seperate concerns to enhance readibility and maintainibility \
                    don't change the exports or imports of the code \
                    insert semicolon where ever is needed but don't and redundant semicolons \
                    Omit the language name at the top of the code \
                    don't change the css classname it could damage the html styling 
                    that is delimited by triple backticks \
                    code: ```{code}``` 
                    {format_instructions}
                          """



# Define response schemas
refactored_code_schema = ResponseSchema(name="refactor_code", description="extract the refactored code")
language_schema = ResponseSchema(name="language",description="Programming language of refactored code")
response_schemas = [refactored_code_schema, language_schema]

# Initialize the output parser
code_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = code_parser.get_format_instructions()
prompt = ChatPromptTemplate.from_template(CODE_TEMPLATE)

# Dependency for Chat Model
def get_chat_instance():
    return chat_model


@app.post("/refactor-code/",status_code=status.HTTP_200_OK) 
async def refactor_code(code: CodeInput, response: Response, chat_ai: ChatOpenAI = Depends(get_chat_instance)) -> Union[ErrorResponse, CodeOutput]:
    try:
        data = prompt.format_messages(code=code.code,format_instructions=format_instructions )
        response_from_ai = chat_ai(data)
        if not response_from_ai.content:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Empty response from chat API")

        refactored_code = code_parser.parse(response_from_ai.content)
        if not refactored_code:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Unexpected result from chat API")
        
        return refactored_code
    
    except HTTPException as e:
        error_response = ErrorResponse(status_code=e.status_code, message=e.detail)
        response.status_code = e.status_code 
        return error_response
    except Exception as e:
        error_message = f"Internal Server error: {e}"
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        error_response = ErrorResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,message=error_message)
        return error_response
    
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
