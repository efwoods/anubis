from urllib.parse import quote
from langgraph_sdk import Auth
from supabase import create_async_client
from fastapi import APIRouter, Depends, Header, HTTPException, Request, Security
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from pydantic import BaseModel

from typing import Optional

import os

from dotenv import load_dotenv

import httpx
from functools import lru_cache
from jose import jwt, JWTError
from fastapi.security import APIKeyHeader

from cachetools import TTLCache
import asyncio

import logging
logger = logging.getLogger(__name__)

load_dotenv()

auth = Auth()

security_route = APIRouter()

security = HTTPBearer()

api_key_scheme = APIKeyHeader(name="API-KEY")

ALGORITHMS = ["RS256"]

DOMAIN = os.getenv("AUTH0_DOMAIN")
CLIENT_ID = os.getenv("AUTH0_CLIENT_ID")
CLIENT_SECRET = os.getenv("AUTH0_CLIENT_SECRET")
AUDIENCE = os.getenv("AUTH0_AUDIENCE")
CONNECTION = os.getenv("AUTH0_CONNECTION", "Username-Password-Authentication")

BASE_AUTH_URL = f"https://{DOMAIN}"

import hashlib, secrets

_api_key_cache: TTLCache = TTLCache(maxsize=1000, ttl=300)
_cache_lock = asyncio.Lock()

def generate_api_key() -> str:
    """Generates a secure, persistent API key."""
    return f"sk-{secrets.token_urlsafe(32)}"

def _hash_key(api_key: str) -> str:
    return hashlib.sha256(api_key.encode()).hexdigest()

# ── Management API token (cached) ──────────────────────────────────────────
_mgmt_token_cache: dict = {"token": None, "expires": 0}
import time
async def _get_mgmt_token(request: Request) -> str:
    """Get a Management API token using client credentials."""
    now = time.monotonic()
    if _mgmt_token_cache['token'] and now < _mgmt_token_cache['expires']:
        return _mgmt_token_cache['token']
    result = await request.app.state.httpx_client.post(f"{BASE_AUTH_URL}/oauth/token", json={
        "grant_type": "client_credentials",
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "audience": f"{BASE_AUTH_URL}/api/v2/",
    })
    result.raise_for_status()
    data = result.json()
    _mgmt_token_cache['token'] = data['access_token']
    _mgmt_token_cache['expires'] = now + data['expires_in'] - 60
    return _mgmt_token_cache['token']


async def _mgmt_headers(request: Request) -> dict:
    access_token = await _get_mgmt_token(request)
    return {"Authorization": f"Bearer {access_token}"}


# utility functions
async def signup_user(email: str, password: str, request:Request, name: Optional[str] = None) -> dict:
    try:
        api_key = generate_api_key()
        api_key_hash = _hash_key(api_key)
        payload = {
            "email": email, 
            "password": password, 
            "connection": CONNECTION,
            "name": name,
            "app_metadata":{
                "api_key": api_key_hash
            }
        }

        headers = await _mgmt_headers(request)

        response = await request.app.state.httpx_client.post(
            f"{BASE_AUTH_URL}/api/v2/users",
            json=payload,
            headers=headers,
        ) 

        response.raise_for_status()
        result = {
            "api_key": api_key,
            "message": "Save this key. This key is shown only once and used for every api request."
            }

        return result
    except Exception as e:
        raise HTTPException(detail=f"Error signing up user: {e}", status_code=response.status_code)

async def logout_user(refresh_token: str, request: Request) -> None:
    response = await request.app.state.httpx_client.post(f"{BASE_AUTH_URL}/oauth/revoke", json={
        "client_id":CLIENT_ID,
        "client_secret": CLIENT_SECRET, 
        "token": refresh_token, 
    })
    return response 

async def login_user(email: str, password: str, request: Request) -> dict:
    """
    Authenticates a user and returns access/id/refresh tokens.
    Requires Resource Owner Password Grant to be enabled.
    """
    try:
        response = await request.app.state.httpx_client.post(f"{BASE_AUTH_URL}/oauth/token", json={
            "grant_type": "password",
            "username": email,
            "password": password,
            "audience": AUDIENCE,
            "scope": "openid profile email offline_access",
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
        })
        return response  # access_token, id_token, refresh_token, expires_in
    except Exception as e:
        raise HTTPException(detail="Error logging in user: {e}", status_code=response.status_code)

async def delete_user(user_id: str, request: Request) -> None:
    headers = await _mgmt_headers(request=request)
    response = await request.app.state.httpx_client.delete(f"{BASE_AUTH_URL}/api/v2/users/{user_id}", 
                            headers = headers)
    
    if response.status_code == 204:
        return {"message":"User deleted"}
    
    raise HTTPException(status_code=response.status_code, detail=response.json())


async def get_user(user_id: str, request: Request) -> dict:
    response = await request.app.state.httpx_client.get(
        f"{BASE_AUTH_URL}/api/v2/users/{user_id}",
        headers = await _mgmt_headers(request=request)
    )
    response.raise_for_status()
    return response.json()



async def send_verification_email(user_id: str, request: Request) -> dict:
    response = await request.app.state.httpx_client.post(
        f"{BASE_AUTH_URL}/api/v2/jobs/verification-email",
        json={"user_id": user_id},
        headers=await _mgmt_headers(request=request),
    )
    if response.status_code >= 400:
        raise HTTPException(
            status_code=response.status_code,
            detail=response.json()
        )
    return response.json()  # returns a job object


# Token Verification

@lru_cache(maxsize=1)
async def _get_jwks(request: Request) -> dict:
    resp = await request.app.state.httpx_client.get(f"https://{DOMAIN}/.well-known/jwks.json")
    resp.raise_for_status()
    return resp.json()

async def verify_token(token: str, request:Request) -> dict:
    """Decodes and validates an Auth0 JWT. Returns the payload."""
    jwks = await _get_jwks(request)
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

async def get_user_with_api_key(api_key: str, request: Request) -> dict | None:
    cache_key = _hash_key(api_key)

    async with _cache_lock:
        if cache_key in _api_key_cache:
            return _api_key_cache[cache_key]
        
    headers = await _mgmt_headers(request)
    result = await request.app.state.httpx_client.get(
        f"{BASE_AUTH_URL}/api/v2/users",
        params={"q": f'app_metadata.api_key:"{cache_key}"', "search_engine":"v3"},
        headers = headers,
    )
    result.raise_for_status()
    users = result.json()
    if not users:
        return None
    user = users[0]

    if user['email_verified'] != True:
        raise HTTPException(detail="Email is not yet verified. Please verify email to continue.", status_code=401)
    
    # if user['app_metadata']['logged_in'] != True:
    #     raise HTTPException(detail="User is not logged in. Please log in to continue.")

    async with _cache_lock:
        _api_key_cache[cache_key] = user
    return user


async def get_current_user(request: Request, api_key: str | None = Depends(api_key_scheme)) -> dict:
    """
    This dependency validates the JWT and returns the payload.
    The 'sub' field in the payload is the Auth0 user_id.
    """
    logger.info("breakpoint")
    if not api_key:
        raise HTTPException(status_code=401, detail="Please send API-KEY in request.")
    
    user = await get_user_with_api_key(api_key, request)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    return user
    # if not res:
        # return None
    
    # try:
    #     token = res.credentials
    #     payload = await verify_token(token, request=request)
    #     return payload  # This contains 'sub', 'email', etc.
    # except Exception as e:
    #     raise HTTPException(
    #         status_code=401, 
    #         detail=f"Invalid or expired token: {str(e)}"
    #     )

# ── Routes ─────────────────────────────────────────────────────────────────
@security_route.post("/signup")
async def signup(body: SignupRequest, request:Request):
        user = await signup_user(body.email, body.password, name=body.name, request=request)
        return user

@security_route.get("/resend_verification_email")
async def resend_verification_email(request:Request, current_user: dict = Depends(get_current_user)):
    # TODO: RATE LIMIT API CALL
    return await send_verification_email(current_user["sub"], request=Request)

@security_route.post("/rotate_api_key")
async def rotate_api_key(request: Request, email: str, password: str):

    result = await login_user(email=email, password=password, request=request)
    id_token = result.json().get('id_token', None)
    if id_token:
        current_user = jwt.get_unverified_claims(id_token)
    else:
        raise HTTPException(detail= "Invalid Credentials.", status_code=401)

    new_key = generate_api_key()
    new_key_hash = _hash_key(new_key)

    headers = await _mgmt_headers(request)
    encoded_id = quote(current_user['sub'], safe="")
    try:
        response = await request.app.state.httpx_client.patch(
            f"{BASE_AUTH_URL}/api/v2/users/{encoded_id}",
            json={"app_metadata": {"api_key":new_key_hash}},
            headers=headers
        )
    except Exception as e:
        raise HTTPException(detail=f"Error patching the new api_key: {e}", status_code=response.status_code)

    async with _cache_lock:
        stale = [k for k, v in _api_key_cache.items() if v['user_id'] == current_user['sub']]
        for k in stale:
            del _api_key_cache[k]

    return {"api_key": new_key, "message": "Save this key. This key is shown only once and used on every api request." }


@security_route.post("/forgot_password")
async def forgot_password(email: str, request: Request, current_user = Depends(get_current_user)):
    headers = await _mgmt_headers(request=request)
    result = await request.app.state.httpx_client.post(
        f"{BASE_AUTH_URL}/dbconnections/change_password",
        json={
            "client_id":  CLIENT_ID,
            "email":      email,
            "connection": CONNECTION,  # e.g. "Username-Password-Authentication"
        },
        headers=headers
    )

    if result.status_code != 200:
        raise HTTPException(status_code=result.status_code, detail=result.json())

    # Always return the same message — don't reveal if email exists
    return {"message": "If that email exists, a password reset link has been sent."}

@security_route.delete("/delete_user")
async def delete_user(request: Request, current_user: dict = Depends(get_current_user)):
    # Optional: ensure users can only delete themselves unless admin
    api_key_hash = current_user['app_metadata']['api_key']
    encoded_id = quote(current_user['user_id'], safe="")
    headers = await _mgmt_headers(request)
    response = await request.app.state.httpx_client.delete(f"{BASE_AUTH_URL}/api/v2/users/{encoded_id}", 
                            headers = headers)
    if response.status_code == 204:
        del _api_key_cache[api_key_hash]
        return {"message":"User deleted"}
    
    raise HTTPException(status_code=response.status_code, detail=response.json())

# @security_route.post("/login")
# async def login(body: LoginRequest, request: Request):
#     try:
#         # returns: access_token, refresh_token, id_token, expires_in
#         response = await login_user(body.email, body.password, request=request)
#         response.raise_for_status()
#         logger.warning(f"response.status_code: {response.status_code}")
#         if response.status_code == 200:
#             data = response.json()
#             logger.warning(f"DATA: {data}")
#             user_info = jwt.get_unverified_claims(data.get('id_token'))
#             logger.warning(f"DATA: {user_info}")
#             logger.warning("XXXXXXXXXXXXXXXXXXXXX UPDATE USER LOGIN")
#             payload = {
#                 "app_metadata": {
#                     "logged_in": True
#                 }
#             }
        
#             logger.warning('update login status breakpoint')
#             # Note: user_id must be URL encoded (e.g., auth0|123 -> auth0%7C123)
#             encoded_id = quote(user_info['sub'], safe="")
#             headers = await _mgmt_headers(request)
#             await request.app.state.httpx_client.patch(
#                 f"{BASE_AUTH_URL}/api/v2/users/{encoded_id}",
#                 json=payload,
#                 headers=headers,
#             )
#             return data
#         else:
#             raise HTTPException(status_code=response.status_code, detail=response.json())
#     except Exception as e:
#         raise HTTPException(status_code=response.status_code, detail=response.json())

# @security_route.get("/get_user_profile")
# async def get_user_profile(request: Request, current_user: dict = Depends(get_current_user)):
    # You don't need to pass user_id in the URL or body; 
    # it is extracted from the token you're wearing!
    # user_id = current_user["user_id"]
    # return {"user_id": user_id}

# @security_route.post("/logout")
# async def logout(body: LogoutRequest, request:Request, current_user: dict = Depends(get_current_user)):

#     response = await logout_user(body.refresh_token, request=request)
#     try:

#         response.raise_for_status()
#         if response.status_code == 200:
#             logger.warning("XXXXXXXXXXXXXXXXXXXXX UPDATE USER LOGIN")
#             payload = {
#                 "user_metadata": {
#                     "logged_in": False
#                 }
#             }
        
#             logger.warning('update login status breakpoint')
#             # Note: user_id must be URL encoded (e.g., auth0|123 -> auth0%7C123)
#             encoded_id = quote(current_user['user_id'], safe="")
#             headers = await _mgmt_headers(request)
#             await request.app.state.httpx_client.patch(
#                 f"{BASE_AUTH_URL}/api/v2/users/{encoded_id}",
#                 json=payload,
#                 headers=headers,
#             )
#         return {"message": "Logged out successfully"}
#     except Exception as e:
#         raise HTTPException(detail = response.json(), status_code=response.status_code)

@auth.authenticate
async def authenticate(request: Request, authorization: str ) -> dict:
    """
    This dependency validates the JWT and returns the payload.
    The 'sub' field in the payload is the Auth0 user_id.
    """
    logger.info("breakpoint")
    if not authorization:
        raise HTTPException(status_code=401, detail="Please send API-KEY as 'Authorization': 'API-KEY' in request header.")
    
    user = await get_user_with_api_key(authorization, request)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    return {"identity": user["user_id"], "metadata": {"user_id": user["user_id"]}}


# Token Authentication
# @auth.authenticate
# async def authenticate(authorization: str | None, request: Request) -> Auth.types.MinimalUserDict:
#     """LangGraph calls this on every request to verify the token."""
#     if not authorization:
#         raise Auth.exceptions.HTTPException(status_code=401, detail="No authorization header")

#     scheme, _, token = authorization.partition(" ")
#     if scheme.lower() != "bearer":
#         raise Auth.exceptions.HTTPException(status_code=401, detail="Invalid auth scheme")

#     try:
#         payload = await verify_token(token, request=request)
#     except Exception as e:
#         raise Auth.exceptions.HTTPException(status_code=401, detail=str(e))

#     # Must return a dict with at least "identity"
#     return {
#         "identity": payload["sub"],          # Auth0 user ID e.g. "auth0|abc123"
#         "email":    payload.get("email"),
#         "permissions": payload.get("permissions", []),
#         "metadata": {"user_id": payload["sub"]}
#     }
