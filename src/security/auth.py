from urllib.parse import quote
from langgraph_sdk import Auth
from supabase import create_async_client
from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from pydantic import BaseModel

from typing import Optional

import os

from dotenv import load_dotenv

import httpx
from functools import lru_cache
from jose import jwt, JWTError

import logging
logger = logging.getLogger(__name__)

load_dotenv()

auth = Auth()

security_route = APIRouter()

security = HTTPBearer()

ALGORITHMS = ["RS256"]

DOMAIN = os.getenv("AUTH0_DOMAIN")
CLIENT_ID = os.getenv("AUTH0_CLIENT_ID")
CLIENT_SECRET = os.getenv("AUTH0_CLIENT_SECRET")
AUDIENCE = os.getenv("AUTH0_AUDIENCE")
CONNECTION = os.getenv("AUTH0_CONNECTION", "Username-Password-Authentication")

BASE_AUTH_URL = f"https://{DOMAIN}"


# ── Management API token (cached) ──────────────────────────────────────────
@lru_cache(maxsize=1)
async def _get_mgmt_token(request: Request) -> str:
    """Get a Management API token using client credentials."""
    resp = await request.app.state.httpx_client.post(f"{BASE_AUTH_URL}/oauth/token", json={
        "grant_type": "client_credentials",
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "audience": f"{BASE_AUTH_URL}/api/v2/",
    })
    resp.raise_for_status()
    return resp.json()["access_token"]


async def _mgmt_headers(request: Request) -> dict:
    access_token = await _get_mgmt_token(request)
    return {"Authorization": f"Bearer {access_token}"}


# utility functions

async def signup_user(email: str, password: str, request:Request, name: Optional[str] = None) -> dict:

    payload = {
        "email": email, 
        "password": password, 
        "connection": CONNECTION,
        "name": name
    }

    headers = await _mgmt_headers(request)


    response = await request.app.state.httpx_client.post(
        f"{BASE_AUTH_URL}/api/v2/users",
        json=payload,
        headers=headers,
    ) 

    return response

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

async def delete_user(user_id: str, request: Request) -> None:
    headers = await _mgmt_headers(request)
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

async def get_current_user(request: Request, res: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> dict:
    """
    This dependency validates the JWT and returns the payload.
    The 'sub' field in the payload is the Auth0 user_id.
    """
    logger.info("breakpoint")
    if not res:
        return None
    
    try:
        token = res.credentials
        payload = await verify_token(token, request=request)
        return payload  # This contains 'sub', 'email', etc.
    except Exception as e:
        raise HTTPException(
            status_code=401, 
            detail=f"Invalid or expired token: {str(e)}"
        )

# ── Routes ─────────────────────────────────────────────────────────────────
@security_route.post("/signup")
async def signup(body: SignupRequest, request:Request):
    response = await signup_user(body.email, body.password, name=body.name, request=request)
    try:
        response.raise_for_status()
        user = response.json()
        return {"user_id": user["user_id"], "email": user["email"]}
    except Exception as e:
        raise HTTPException(status_code=response.status_code, detail=response.json())
    
@security_route.post("/resend-verification")
async def resend_verification(request:Request, current_user: dict = Depends(get_current_user)):
    return await send_verification_email(current_user["sub"], request=Request)

@security_route.delete("/users/{user_id}")
def delete(user_id: str, current_user: dict = Depends(get_current_user)):
    # Optional: ensure users can only delete themselves unless admin
    encoded_id = quote(user_id, safe="")
    if current_user["sub"] != user_id:
        raise HTTPException(status_code=403, detail="Forbidden")
    response = delete_user(encoded_id)
    return response
   
@security_route.post("/login")
async def login(body: LoginRequest, request: Request):
    try:
        # returns: access_token, refresh_token, id_token, expires_in
        response = await login_user(body.email, body.password, request=request)
        response.raise_for_status()
        logger.warning(f"response.status_code: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            logger.warning(f"DATA: {data}")
            user_info = jwt.get_unverified_claims(data.get('id_token'))
            logger.warning(f"DATA: {user_info}")
            logger.warning("XXXXXXXXXXXXXXXXXXXXX UPDATE USER LOGIN")
            payload = {
                "user_metadata": {
                    "logged_in": True
                }
            }
        
            logger.warning('update login status breakpoint')
            # Note: user_id must be URL encoded (e.g., auth0|123 -> auth0%7C123)
            encoded_id = quote(user_info['sub'], safe="")
            headers = await _mgmt_headers(request)
            await request.app.state.httpx_client.patch(
                f"{BASE_AUTH_URL}/api/v2/users/{encoded_id}",
                json=payload,
                headers=headers,
            )
            return data
        else:
            raise HTTPException(status_code=response.status_code, detail=response.json())
    except Exception as e:
        raise HTTPException(status_code=response.status_code, detail=response.json())

@security_route.get("/get_user_profile")
async def get_user_profile(request: Request, current_user: dict = Depends(get_current_user)):
    # You don't need to pass user_id in the URL or body; 
    # it is extracted from the token you're wearing!
    user_id = current_user["sub"]
    response = await get_user(request=request, user_id=user_id)
    return {"user_id": user_id, "full_payload": response}

@security_route.post("/logout")
async def logout(body: LogoutRequest, request:Request, current_user: dict = Depends(get_current_user)):

    response = await logout_user(body.refresh_token, request=request)
    try:

        response.raise_for_status()
        if response.status_code == 200:
            logger.warning("XXXXXXXXXXXXXXXXXXXXX UPDATE USER LOGIN")
            payload = {
                "user_metadata": {
                    "logged_in": False
                }
            }
        
            logger.warning('update login status breakpoint')
            # Note: user_id must be URL encoded (e.g., auth0|123 -> auth0%7C123)
            encoded_id = quote(current_user['sub'], safe="")
            headers = await _mgmt_headers(request)
            await request.app.state.httpx_client.patch(
                f"{BASE_AUTH_URL}/api/v2/users/{encoded_id}",
                json=payload,
                headers=headers,
            )
        return {"message": "Logged out successfully"}
    except Exception as e:
        raise HTTPException(detail = response.json(), status_code=response.status_code)


@auth.authenticate
async def authenticate(authorization: str | None) -> Auth.types.MinimalUserDict:
    """LangGraph calls this on every request to verify the token."""
    if not authorization:
        raise Auth.exceptions.HTTPException(status_code=401, detail="No authorization header")

    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer":
        raise Auth.exceptions.HTTPException(status_code=401, detail="Invalid auth scheme")

    try:
        payload = await verify_token(token)
    except Exception as e:
        raise Auth.exceptions.HTTPException(status_code=401, detail=str(e))

    # Must return a dict with at least "identity"
    return {
        "identity": payload["sub"],          # Auth0 user ID e.g. "auth0|abc123"
        "email":    payload.get("email"),
        "permissions": payload.get("permissions", []),
        "metadata": {"user_id": payload["sub"]}
    }
