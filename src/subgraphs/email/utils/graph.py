from langgraph.graph import StateGraph, State, END
from src.anubis.utils.context import GlobalContext
from src.anubis.utils.state import GlobalState
from src.anubis.utils.nodes import load_consciousness
from src.anubis.utils.model import init_model

""" Purpose """
# Automated cron query api, email forwarding or ingestion to the endpoint
# API to configure the cron and select the assistant for response

# Accept Email: configured assistant and current user
# Triage for ignore, notify, and respond
# Load the response model with the load conscioussness endpoint
# Prompt the model to write a response if there is a response chosen
# Ask for acceptance or to edit the response with an interrupt
# respond

""" Nodes """
# async def accept_email(state: EmailState, config: RunnableConfig, runtime: Runtime[GlobalContext]) -> EmailState:
#     """ Endpoint to accept an email to be triaged and responded to. """


# async def classify_next_action(state: EmailState, config: RunnableConfig, runtime: Runtime[GlobalContext]):
#     """ Ignore, notify, or respond to the email """
#     from pydantic import BaseModel, Field
#     class classifyEmail(BaseModel):
#         """ Classify an Email to Determine whether to ignore the email, notify the user to ask for further direction, or respond to the email. """

# async def write_response(state: EmailState, config: RunnableConfig, runtime: Runtime[GlobalContext]):
#     """ Write a custom response """

# """ Graph """
# email_processing_workflow = StateGraph(

# )
