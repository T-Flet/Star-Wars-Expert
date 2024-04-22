# Chains
from langchain_core.pydantic_v1 import BaseModel, Field

# To serve the app
from fastapi import FastAPI
from langchain_core.messages import BaseMessage
from langserve import add_routes, CustomUserType

import dotenv
dotenv.load_dotenv()

from ingredients import script_db, woo_db, full_chain, compound_chain, agent_executor



## Type specifications (with unusual class-scope fields)

class StrInput(BaseModel):
    input: str

class Input(BaseModel):
    input: str
    chat_history: list[BaseMessage] = Field(
        ...,
        extra = dict(widget = dict(type = 'chat', input = 'location')),
    )

class Output(BaseModel):
    output: str



## App definition
# NOTE: The chat playground type has a web page issue (flashes and becomes white, hence non-interactable; this was supposedly solved in an issue late last year)

app = FastAPI(
    title = 'Star Wars Expert',
    version = '1.0',
    description = 'A Star Wars expert chatbot',
)


# Basic retriever versions

# add_routes(app, script_db.as_retriever())
# add_routes(app, woo_db.as_retriever())


# History-aware retriever version
# add_routes(app, full_chain.with_types(input_type = StrInput, output_type = Output), playground_type = 'default')


# Agent version

# add_routes(app, agent_executor, playground_type = 'chat')
# add_routes(app, agent_executor.with_types(input_type = StrInput, output_type = Output))


# Non-agent chain-logic version

add_routes(app, compound_chain.with_types(input_type = StrInput))
# add_routes(app, compound_chain.with_types(input_type = Input), playground_type = 'chat')



if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host = 'localhost', port = 8000)


