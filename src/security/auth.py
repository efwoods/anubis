from langgraph_sdk import Auth
from supabase import create_async_client
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse, HTTPException

from src.api.dependencies import (
    get_context, 
    get_request_object, 
)

auth = Auth()

security_route = APIRouter()

@security_route.post("/sign_up")
async def sign_up(email: str, password: str, context=Depends(get_context)):

    try:
        supabase_client = create_async_client(supabase_url=context.supabase_url, supabase_key=context.supabase_key)
        response = await supabase_client.auth.signup(
            {"email": email, 
             "password": password}
        )

        return JSONResponse(response, static_code = 200)
    except Exception as e:
        return HTTPException(status_code=500, detail=f"Error: {e}")

@security_route.post("/login")
async def login(email: str, password: str, context=Depends(get_context, get_request_object)):
    try:
        supabase_client = create_async_client(supabase_url=context.supabase_url, supabase_key=context.supabase_key)
        response = await supabase_client.auth.sign_in_with_password(
            {
                "email": email, 
                "password": password
            }
        )
        return JSONResponse(response, status_code=200)
    except Exception as e:
        return HTTPException(status_code=500, detail=f"Error: {e}")

@security_route.get("get_current_user")
@auth.authenticate
async def get_current_user(authorization: str | None) -> Auth.types.MinimalUserDict:
    """ Identify if the user's token is valid. """
    assert authorization
    scheme, token = authorization.split()
    assert scheme.lower() == "bearer"

    # Check if token is valid:
    if token




