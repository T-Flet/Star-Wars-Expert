from langchain_openai import ChatOpenAI#, OpenAIEmbeddings # No need to pay for using embeddings as well when have free alternatives

# Data
from langchain_community.document_loaders import DirectoryLoader, TextLoader, WebBaseLoader
# from langchain_chroma import Chroma # The documentation uses this one, but it is extremely recent, and the same functionality is available in langchain_community and langchain (which imports community)
from langchain_community.vectorstores import Chroma # This has documentation on-hover, while the indirect import through non-community does not
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings # The free alternative (also the default in docs, with model_name = 'all-MiniLM-L6-v2')
from langchain.text_splitter import RecursiveCharacterTextSplitter # Recursive to better keep related bits contiguous (also recommended in docs: https://python.langchain.com/docs/modules/data_connection/document_transformers/)

# Chains
from langchain.prompts import PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.tools.retriever import create_retriever_tool
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, chain
from langchain_core.pydantic_v1 import BaseModel, Field

# Agents
from langchain import hub
from langchain.agents import create_tool_calling_agent, AgentExecutor

# To manually create inputs to test pipelines
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document

import requests
from bs4 import BeautifulSoup

import os
import shutil
from pathlib import Path
import re

import dotenv
dotenv.load_dotenv()




## Vector stores

# Non-persistent; build from documents

# scripts = DirectoryLoader('scripts', glob = '*.txt', loader_cls = TextLoader).load()
# for s in scripts: s.page_content = re.sub(r'^[\t ]+', '', s.page_content, flags = re.MULTILINE)  # Spacing to centre text noise
# script_chunks = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200, separators = ['\n\n\n', '\n\n', '\n']).split_documents(scripts)
# script_db = Chroma.from_documents(script_chunks, SentenceTransformerEmbeddings(model_name = 'all-MiniLM-L6-v2'))

# pages = DirectoryLoader('wookieepedia', glob = '*.txt', loader_cls = TextLoader).load()
# page_chunks = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200, separators = ['\n\n\n', '\n\n', '\n']).split_documents(pages)
# woo_db = Chroma.from_documents(page_chunks, SentenceTransformerEmbeddings(model_name = 'all-MiniLM-L6-v2'))


# # Load pre-built persistent ones

script_db = Chroma(embedding_function = SentenceTransformerEmbeddings(model_name = 'all-MiniLM-L6-v2'), persist_directory = str(Path('scripts') / 'db'))
woo_db = Chroma(embedding_function = SentenceTransformerEmbeddings(model_name = 'all-MiniLM-L6-v2'), persist_directory = str(Path('wookieepedia') / 'db'))



# Chains

llm = ChatOpenAI(model = 'gpt-3.5-turbo-0125', temperature = 0)


## Base version (only one retriever)

document_prompt_system_text = '''
You are very knowledgeable about Star Wars and your job is to answer questions about its plot, characters, etc.
Use the context below to produce your answers with as much detail as possible.
If you do not know an answer, say so; do not make up information not in the context.

<context>
{context}
</context>
'''
document_prompt = ChatPromptTemplate.from_messages([
    ('system', document_prompt_system_text),
    MessagesPlaceholder(variable_name = 'chat_history', optional = True),
    ('user', '{input}')
])
document_chain = create_stuff_documents_chain(llm, document_prompt)

script_retriever_prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name = 'chat_history'),
    ('user', '{input}'),
    ('user', '''Given the above conversation, generate a search query to look up relevant information in a database containing the full scripts from the Star Wars films (i.e. just dialogue and brief scene descriptions).
     The query need not be a proper sentence, but a list of keywords likely to be in dialogue or scene descriptions''')
])
script_retriever_chain = create_history_aware_retriever(llm, script_db.as_retriever(), script_retriever_prompt) # Essentially just: prompt | llm | StrOutputParser() | retriever

woo_retriever_prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name = 'chat_history'),
    ('user', '{input}'),
    ('user', 'Given the above conversation, generate a search query to find a relevant page in the Star Wars fandom wiki; the query should be something simple, such as the name of a character, place, event, item, etc.')
])
woo_retriever_chain = create_history_aware_retriever(llm, woo_db.as_retriever(), woo_retriever_prompt) # Essentially just: prompt | llm | StrOutputParser() | retriever

# full_chain = create_retrieval_chain(script_retriever_chain, document_chain)
full_chain = create_retrieval_chain(woo_retriever_chain, document_chain)



# simplify_query_prompt = ChatPromptTemplate.from_messages([
#     ('system', 'Given the above conversation, generate a search query to find a relevant page in the Star Wars fandom wiki; the query should be something simple, at most 4 words, such as the name of a character, place, event, item, etc.'),
#     MessagesPlaceholder('chat_history', optional = True), # Using this form since not clear how to have optional = True in the tuple form
#     ('human', '{query}')
# ])

# simplify_query_chain = simplify_query_prompt | llm | StrOutputParser() # To extract just the message



## Agent version

script_tool = create_retriever_tool(
    script_db.as_retriever(search_kwargs = dict(k = 4)),
    'search_film_scripts',
    '''Search the Star Wars film scripts. This tool should be the first choice for Star Wars related questions.
    Queries passed to this tool should be lists of keywords likely to be in dialogue or scene descriptions, and should not include film titles.'''
)

woo_tool = create_retriever_tool(
    woo_db.as_retriever(search_kwargs = dict(k = 4)),
    'search_wookieepedia',
    'Search the Star Wars fandom wiki. This tool should be the first choice for Star Wars related questions.'
    # This tool should be used for queries about details of a particular character, location, event, weapon, etc., and the query should be something simple, such as the name of a character, place, event, item, etc.'''
)
tools = [script_tool, woo_tool]

agent_system_text = '''
You are a helpful agent who is very knowledgeable about Star Wars and your job is to answer questions about its plot, characters, etc.
Use the context provided in the exchanges to come to produce your answers with as much detail as possible.
If you do not know an answer, say so; do not make up information.
'''
agent_prompt = ChatPromptTemplate.from_messages([
    ('system', agent_system_text),
    MessagesPlaceholder('chat_history', optional = True), # Using this form since not clear how to have optional = True in the tuple form
    ('human', '{input}'),
    ('placeholder', '{agent_scratchpad}') # Required for chat history and the agent's intermediate processing values
])
agent = create_tool_calling_agent(llm, tools, agent_prompt)

agent_executor = AgentExecutor(agent = agent, tools = tools, verbose = True)



## Non-agent chain-logic version

# Determine which retriever is best and generate an appropriate query for it
class DirectedQuery(BaseModel):
    '''Determine whether a query is best answered by looking at scripts rather than articles'''

    query: str = Field(
        ...,
        description = '''The query to either search film scripts or wiki articles.
        A film script query should include character names and relevant keywords of what they are saying in the a scene which is likely to contain the required information.
        A wiki articles search should instead be at most 4 words, simply being the name of a character or location or event whose page is likely to contain the required information.''',
    )
    source: str = Field(
        ...,
        description = 'Either "wiki" or "scripts", indicating which source the query should be passed to.',
    )
query_analyser_prompt = ChatPromptTemplate.from_messages([
        ('system', 'You have the ability to issue search queries of one of two kinds to get information to help answer questions.'),
        ('human', '{question}'),
])
structured_llm = llm.with_structured_output(DirectedQuery)
query_generator = dict(question = RunnablePassthrough()) | query_analyser_prompt | structured_llm

retrievers = dict(wiki = woo_db.as_retriever(search_kwargs = dict(k = 4)), scripts = script_db.as_retriever(search_kwargs = dict(k = 4)))

@chain
def compound_retriever(question):
    response = query_generator.invoke(question)
    retriever = retrievers[response.source]
    return retriever.invoke(response.query)

compound_chain = create_retrieval_chain(compound_retriever, document_chain)



## Wookieepedia functions

def first_wookieepedia_result(query: str) -> str:
    '''Get the url of the first result when searching Wookieepedia for a query
    (best for simple names as queries, ideally generated by the llm for something like
    "Produce a input consisting of the name of the most important element in the query so that its article can be looked up")
    '''
    search_results = requests.get(f'https://starwars.fandom.com/wiki/Special:Search?query={"+".join(query.split(" "))}')
    soup = BeautifulSoup(search_results.content, 'html.parser')
    first_res = soup.find('a', class_ = 'unified-search__result__link')
    return first_res['href']


def get_wookieepedia_page_content(query: str, previous_sources: set[str]) -> Document | None:
    '''Return cleaned content from a Wookieepedia page provided it was not already sourced
    '''
    url = first_wookieepedia_result(query)

    if url in previous_sources: return None
    else:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        doc = soup.find('div', id = 'content').get_text()

        # Cleaning
        doc = doc.split('\n\n\n\n\n\n\n\n\n\n\n\n\n\n')[-1] # The (multiple) preambles are separated by these many newlines; no harm done if not present
        doc = re.sub('\[\d*\]', '', doc) # References (and section title's "[]" suffixes) are noise
        doc = doc.split('\nAppearances\n')[0] # Keep only content before these sections
        doc = doc.split('\nSources\n')[0] # Technically no need to check this if successfully cut on appearances, but no harm done
        doc = re.sub('Contents\n\n(?:[\d\.]+ [^\n]+\n+)+', '', doc) # Remove table of contents

        return Document(page_content = doc, metadata = dict(source = url))

def get_wookieepedia_context(original_query: str, simple_query: str, wdb: Chroma) -> list[Document]:
    try:
        doc = get_wookieepedia_page_content(simple_query, previous_sources = set(md.get('source') for md in wdb.get()['metadatas']))
        if doc is not None:
            new_chunks = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200).split_documents([doc])
            wdb.add_documents(new_chunks)
            print(f"Added new chunks (for '{simple_query}' -> {doc['metadata']['source']}) to the Wookieepedia database.")
    except: return []

    return wdb.similarity_search(original_query, k = 10)


