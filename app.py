from langchain_core.messages import HumanMessage, AIMessage

import gradio as gr

# For local testing; not used in the Huggingface space
import dotenv
dotenv.load_dotenv()

# The available backends to use in the app
from ingredients import script_db, woo_db, full_chain, compound_chain, agent_executor


def chat(message, history):
    formatted_history = []
    for human, ai in history:
        formatted_history.append(HumanMessage(content = human))
        formatted_history.append(AIMessage(content = ai))
    
    # Yes, the context chat entries are not fed back to the system, but that is probably for the best due to input size limit
    response = compound_chain.invoke(dict(input = HumanMessage(content = message), chat_history = formatted_history))

    return response['answer']


gr.ChatInterface(
    chat,
    textbox = gr.Textbox(placeholder = 'Ask something about Star Wars', container = False, scale = 7),
    title = 'Star Wars Expert', description = 'I am knowledgeable about Star Wars; ask me about it',
    examples = ['Do you know the tragedy of Darth Plagueis the Wise?', 'What power source did the Death Star use?', "Who participates in Han's rescue from Jabba? And where is the palace?"],
    cache_examples = False, # This would avoid invoking the chatbot for the example queries (it would invokes it on them on startup instead)
    theme = 'soft', retry_btn = None, undo_btn = 'Delete Previous', clear_btn = 'Clear'
).launch()


