from urllib.parse import quote
from langgraph_sdk import Auth
from supabase import create_async_client
from fastapi import APIRouter, Depends, Header, HTTPException
from fastapi.responses import JSONResponse

from pydantic import BaseModel

from typing import Optional

from src.api.dependencies import (
    get_context, 
    get_request_object, 
)

import os

from dotenv import load_dotenv

import httpx
from functools import lru_cache
from jose import jwt, JWTError

load_dotenv()

auth = Auth()

security_route = APIRouter()

ALGORITHMS = ["RS256"]

DOMAIN = os.getenv("AUTH0_DOMAIN")
CLIENT_ID = os.getenv("AUTH0_CLIENT_ID")
CLIENT_SECRET = os.getenv("AUTH0_CLIENT_SECRET")
AUDIENCE = os.getenv("AUTH0_AUDIENCE")
CONNECTION = os.getenv("AUTH0_CONNECTION", "Username-Password-Authentication")

BASE_AUTH_URL = f"https://{DOMAIN}"


# ── Management API token (cached) ──────────────────────────────────────────
@lru_cache(maxsize=1)
def _get_mgmt_token() -> str:
    """Get a Management API token using client credentials."""
    resp = httpx.post(f"{BASE_AUTH_URL}/oauth/token", json={
        "grant_type": "client_credentials",
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "audience": f"{BASE_AUTH_URL}/api/v2/",
    })
    resp.raise_for_status()
    return resp.json()["access_token"]


def _mgmt_headers() -> dict:
    return {"Authorization": f"Bearer {_get_mgmt_token()}"}


# utility functions

def signup_user(email: str, password: str, name: Optional[str] = None) -> dict:

    payload = {
        "email": email, 
        "password": password, 
        "connection": CONNECTION,
        "name": name
    }

    response = httpx.post(
        f"{BASE_AUTH_URL}/api/v2/users",
        json=payload,
        headers=_mgmt_headers(),
    ) 

    return response

def logout_user(refresh_token: str) -> None:
    response = httpx.post(f"{BASE_AUTH_URL}/oauth/revoke", json={
        "client_id":CLIENT_ID,
        "client_secret": CLIENT_SECRET, 
        "token": refresh_token, 
    })
    return response 

def login_user(email: str, password: str) -> dict:
    """
    Authenticates a user and returns access/id/refresh tokens.
    Requires Resource Owner Password Grant to be enabled.
    """
    response = httpx.post(f"{BASE_AUTH_URL}/oauth/token", json={
        "grant_type": "password",
        "username": email,
        "password": password,
        "audience": AUDIENCE,
        "scope": "openid profile email offline_access",
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
    })
    return response  # access_token, id_token, refresh_token, expires_in

def delete_user(user_id: str) -> None:
    response = httpx.delete(f"{BASE_AUTH_URL}/api/v2/users/{user_id}", 
                            headers = _mgmt_headers())
    
    if response.status_code == 204:
        return {"message":"User deleted"}
    
    raise HTTPException(status_code=response.status_code, detail=response.json())

    return response.json()

def get_user(user_id: str) -> dict:
    response = httpx.get(
        f"{BASE_AUTH_URL}/api/v2/users/{user_id}",
        headers = _mgmt_headers()
    )
    response.raise_for_status()
    return response.json()

# Token Verification

@lru_cache(maxsize=1)
def _get_jwks() -> dict:
    resp = httpx.get(f"https://{DOMAIN}/.well-known/jwks.json")
    resp.raise_for_status()
    return resp.json()

def verify_token(token: str) -> dict:
    """Decodes and validates an Auth0 JWT. Returns the payload."""
    jwks = _get_jwks()
    unverified_header = jwt.get_unverified_header(token)
    
    rsa_key = next(
        (
            {
                "kty": key["kty"], "kid": key["kid"],
                "use": key["use"], "n": key["n"], "e": key["e"],
            }
            for key in jwks["keys"]
            if key["kid"] == unverified_header["kid"]
        ),
        None,
    )
    if not rsa_key:
        raise JWTError("Unable to find matching key")

    return jwt.decode(
        token,
        rsa_key,
        algorithms=ALGORITHMS,
        audience=AUDIENCE,
        issuer=f"https://{DOMAIN}/",
    )

# ── Schemas ────────────────────────────────────────────────────────────────
class SignupRequest(BaseModel):
    email: str
    password: str
    name: str | None = None

class LoginRequest(BaseModel):
    email: str
    password: str

class LogoutRequest(BaseModel):
    refresh_token: str

# ── Dependency: require valid token ────────────────────────────────────────
def get_current_user(authorization: str = Header(...)) -> dict:
    token = authorization.removeprefix("Bearer ")
    try:
        return verify_token(token)
    except Exception as e:
        raise HTTPException(status_code=401, detail=str(e))

# ── Routes ─────────────────────────────────────────────────────────────────



@security_route.post("/signup")
def signup(body: SignupRequest):
    response = signup_user(body.email, body.password, name=body.name)
    try:
        response.raise_for_status()
        user = response.json()
        return {"user_id": user["user_id"], "email": user["email"]}
    except Exception as e:
        raise HTTPException(status_code=response.status_code, detail=response.json())
    

@security_route.post("/login")
def login(body: LoginRequest):
    try:
        # returns: access_token, refresh_token, id_token, expires_in
        response = login_user(body.email, body.password)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        raise HTTPException(status_code=response.status_code, detail=response.json())

@security_route.post("/logout")
def logout(body: LogoutRequest):

    response = logout_user(body.refresh_token)
    try:
        response.raise_for_status()
        return {"message": "Logged out successfully"}
    except Exception as e:
        raise HTTPException(detail = response.json(), status_code=response.status_code)

@security_route.delete("/users/{user_id}")
def delete(user_id: str, current_user: dict = Depends(get_current_user)):
    # Optional: ensure users can only delete themselves unless admin
    encoded_id = quote(user_id, safe="")
    if current_user["sub"] != user_id:
        raise HTTPException(status_code=403, detail="Forbidden")
    response = delete_user(encoded_id)
    return response

# ###

# @security_route.post("/sign_up")
# async def sign_up(email: str, password: str, context=Depends(get_context)):

#     try:
#         supabase_client = create_async_client(supabase_url=context.supabase_url, supabase_key=context.supabase_key)
#         response = await supabase_client.auth.signup(
#             {"email": email, 
#              "password": password}
#         )

#         return JSONResponse(response, static_code = 200)
#     except Exception as e:
#         return HTTPException(status_code=500, detail=f"Error: {e}")

# @security_route.post("/login")
# async def login(email: str, password: str, context=Depends(get_context, get_request_object)):
#     try:
#         supabase_client = create_async_client(supabase_url=context.supabase_url, supabase_key=context.supabase_key)
#         response = await supabase_client.auth.sign_in_with_password(
#             {
#                 "email": email, 
#                 "password": password
#             }
#         )
#         return JSONResponse(response, status_code=200)
#     except Exception as e:
#         return HTTPException(status_code=500, detail=f"Error: {e}")




# @security_route.get("get_current_user")
# @auth.authenticate
# async def get_current_user(authorization: str | None) -> Auth.types.MinimalUserDict:
#     """ Identify if the user's token is valid. """
#     assert authorization
#     scheme, token = authorization.split()
#     assert scheme.lower() == "bearer"

#     # Check if token is valid:
#     if token

@auth.authenticate
async def authenticate(authorization: str | None) -> Auth.types.MinimalUserDict:
    """LangGraph calls this on every request to verify the token."""
    if not authorization:
        raise Auth.exceptions.HTTPException(status_code=401, detail="No authorization header")

    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer":
        raise Auth.exceptions.HTTPException(status_code=401, detail="Invalid auth scheme")

    try:
        payload = verify_token(token)
    except Exception as e:
        raise Auth.exceptions.HTTPException(status_code=401, detail=str(e))

    # Must return a dict with at least "identity"
    return {
        "identity": payload["sub"],          # Auth0 user ID e.g. "auth0|abc123"
        "email":    payload.get("email"),
        "permissions": payload.get("permissions", []),
    }

