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


def update_user_login_status(user_id: str, is_logged_in: bool):
    """Updates Auth0 user_metadata with login status."""
    payload = {
        "user_metadata": {
            "logged_in": is_logged_in
        }
    }

    logger.warning('update login status breakpoint')
    # Note: user_id must be URL encoded (e.g., auth0|123 -> auth0%7C123)
    encoded_id = quote(user_id, safe="")
    httpx.patch(
        f"{BASE_AUTH_URL}/api/v2/users/{encoded_id}",
        json=payload,
        headers=_mgmt_headers(),
    )

def get_user(user_id: str) -> dict:
    response = httpx.get(
        f"{BASE_AUTH_URL}/api/v2/users/{user_id}",
        headers = _mgmt_headers()
    )
    response.raise_for_status()
    return response.json()



def send_verification_email(user_id: str) -> dict:
    response = httpx.post(
        f"{BASE_AUTH_URL}/api/v2/jobs/verification-email",
        json={"user_id": user_id},
        headers=_mgmt_headers(),
    )
    if response.status_code >= 400:
        raise HTTPException(
            status_code=response.status_code,
            detail=response.json()
        )
    return response.json()  # returns a job object


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

async def get_current_user(request: Request, res: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> dict:
    """
    This dependency validates the JWT and returns the payload.
    The 'sub' field in the payload is the Auth0 user_id.
    """
    if not res:
        return None
    
    try:
        token = res.credentials
        payload = verify_token(token)
        return payload  # This contains 'sub', 'email', etc.
    except Exception as e:
        raise HTTPException(
            status_code=401, 
            detail=f"Invalid or expired token: {str(e)}"
        )

# ── Routes ─────────────────────────────────────────────────────────────────

@security_route.post("/resend-verification")
def resend_verification(current_user: dict = Depends(get_current_user)):
    return send_verification_email(current_user["sub"])

@security_route.get("/get_user_profile")
def get_user_profile(current_user: dict = Depends(get_current_user)):
    # You don't need to pass user_id in the URL or body; 
    # it is extracted from the token you're wearing!
    user_id = current_user["sub"]
    response = get_user(user_id=user_id)
    return {"user_id": user_id, "full_payload": response}

@security_route.post("/signup")
def signup(body: SignupRequest):
    response = signup_user(body.email, body.password, name=body.name)
    try:
        response.raise_for_status()
        user = response.json()
        return {"user_id": user["user_id"], "email": user["email"]}
    except Exception as e:
        raise HTTPException(status_code=response.status_code, detail=response.json())
    
import logging
logger = logging.getLogger(__name__)
@security_route.post("/login")
def login(body: LoginRequest):
    try:
        # returns: access_token, refresh_token, id_token, expires_in
        response = login_user(body.email, body.password)
        response.raise_for_status()
        logger.warning(f"response.status_code: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            logger.warning(f"DATA: {data}")
            user_info = jwt.get_unverified_claims(data.get('id_token'))
            logger.warning(f"DATA: {user_info}")
            update_user_login_status(user_info['sub'], True)
            return data
        else:
            raise HTTPException(status_code=response.status_code, detail=response.json())
    except Exception as e:
        raise HTTPException(status_code=response.status_code, detail=response.json())

@security_route.post("/logout")
def logout(body: LogoutRequest, current_user: dict = Depends(get_current_user)):

    response = logout_user(body.refresh_token)
    try:

        response.raise_for_status()
        if response.status_code == 200:
            update_user_login_status(current_user['sub'], False)
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
        "metadata": {"user_id": payload["sub"]}
    }
