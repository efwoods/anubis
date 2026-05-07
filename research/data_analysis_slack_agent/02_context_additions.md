# Additions to `GlobalContext`

Per the workspace rule (`.cursorrules`):

> whenever there is a standard environment variable for settings, that needs to be set in the env and dev env with uppercase and included in context in lowercase (following field format for strings with descriptions and int and floats with default values as per the other variables in context.py) and the context needs to be called at the top of the function and passed throughout nested calls to access environment variables.

Below are the **field declarations** to add to `@dataclass(kw_only=True) class GlobalContext` in `src/anubis/utils/context.py`. They follow the exact format used by existing fields like `openai_api_key`, `whisper_max_bytes`, etc.

```python
""" <Data Analysis Agent — Sandbox> """

sandbox_provider: str = field(
    default="local",
    metadata={
        "description": "Backend for deepagents code execution. One of: local | daytona | modal | runloop | agentcore. Env SANDBOX_PROVIDER."
    },
)

sandbox_root_dir: str = field(
    default="/tmp/anubis_sandbox",
    metadata={
        "description": "Local FS root for LocalShellBackend. Env SANDBOX_ROOT_DIR."
    },
)

daytona_api_key: str = field(
    default=None,
    metadata={"description": "Daytona API key. Env DAYTONA_API_KEY."},
)

modal_token_id: str = field(
    default=None,
    metadata={"description": "Modal token id. Env MODAL_TOKEN_ID."},
)

modal_token_secret: str = field(
    default=None,
    metadata={"description": "Modal token secret. Env MODAL_TOKEN_SECRET."},
)

runloop_api_key: str = field(
    default=None,
    metadata={"description": "Runloop API key. Env RUNLOOP_API_KEY."},
)

""" </Data Analysis Agent — Sandbox> """

""" <Data Analysis Agent — Slack> """

slack_bot_token: str = field(
    default=None,
    metadata={
        "description": "Slack Bot User OAuth token (xoxb-...). Used by slack_sdk.WebClient. Env SLACK_BOT_TOKEN."
    },
)

slack_user_token: str = field(
    default=None,
    metadata={
        "description": "Slack User OAuth token (xoxp-...) for history scopes when bot scopes are insufficient. Env SLACK_USER_TOKEN."
    },
)

slack_signing_secret: str = field(
    default=None,
    metadata={
        "description": "Slack signing secret used to verify incoming Events API requests. Env SLACK_SIGNING_SECRET."
    },
)

slack_app_token: str = field(
    default=None,
    metadata={
        "description": "Slack App-Level token (xapp-...) for Socket Mode in dev. Env SLACK_APP_TOKEN."
    },
)

slack_default_channel: str = field(
    default=None,
    metadata={
        "description": "Channel id (C0123...) the AAR is posted to when no channel is specified. Env SLACK_DEFAULT_CHANNEL."
    },
)

slack_history_max_pages: int = field(
    default=10,
    metadata={
        "description": "Max paginated calls to conversations.history per ingest run. Env SLACK_HISTORY_MAX_PAGES."
    },
)

""" </Data Analysis Agent — Slack> """

""" <Data Analysis Agent — Health/Sleep> """

sleep_provider: str = field(
    default="apple",
    metadata={
        "description": "Sleep data provider. One of: apple | oura | whoop | fitbit. Env SLEEP_PROVIDER."
    },
)

oura_api_token: str = field(
    default=None,
    metadata={"description": "Oura personal access token. Env OURA_API_TOKEN."},
)

whoop_client_id: str = field(
    default=None,
    metadata={"description": "Whoop OAuth client id. Env WHOOP_CLIENT_ID."},
)

whoop_client_secret: str = field(
    default=None,
    metadata={"description": "Whoop OAuth client secret. Env WHOOP_CLIENT_SECRET."},
)

fitbit_client_id: str = field(
    default=None,
    metadata={"description": "Fitbit OAuth client id. Env FITBIT_CLIENT_ID."},
)

fitbit_client_secret: str = field(
    default=None,
    metadata={"description": "Fitbit OAuth client secret. Env FITBIT_CLIENT_SECRET."},
)

""" </Data Analysis Agent — Health/Sleep> """

""" <Data Analysis Agent — Bank of America via Plaid> """

plaid_client_id: str = field(
    default=None,
    metadata={"description": "Plaid client id. Env PLAID_CLIENT_ID."},
)

plaid_secret: str = field(
    default=None,
    metadata={"description": "Plaid secret (env-specific). Env PLAID_SECRET."},
)

plaid_env: str = field(
    default="sandbox",
    metadata={
        "description": "Plaid environment. One of: sandbox | development | production. Env PLAID_ENV."
    },
)

plaid_boa_access_token: str = field(
    default=None,
    metadata={
        "description": "Plaid access_token for the linked Bank of America item. Env PLAID_BOA_ACCESS_TOKEN."
    },
)

""" </Data Analysis Agent — Bank of America via Plaid> """

""" <Data Analysis Agent — Google> """

google_oauth_client_secret_json: str = field(
    default=None,
    metadata={
        "description": "Path to OAuth client_secret.json for Google APIs. Env GOOGLE_OAUTH_CLIENT_SECRET_JSON."
    },
)

google_token_json_path: str = field(
    default=None,
    metadata={
        "description": "Path to persisted user token.json from Google OAuth. Env GOOGLE_TOKEN_JSON_PATH."
    },
)

google_meet_scopes: str = field(
    default="https://www.googleapis.com/auth/meetings.space.readonly,https://www.googleapis.com/auth/drive.readonly,https://www.googleapis.com/auth/documents.readonly",
    metadata={
        "description": "Comma-separated Google API scopes for Meet+Drive+Docs ingestion. Env GOOGLE_MEET_SCOPES."
    },
)

""" </Data Analysis Agent — Google> """

""" <Data Analysis Agent — Git/GitHub> """

github_pat: str = field(
    default=None,
    metadata={
        "description": "GitHub fine-grained PAT with repo + read:org scopes. Env GITHUB_PAT."
    },
)

github_org: str = field(
    default=None,
    metadata={"description": "Default GitHub org for org-wide commit ingest. Env GITHUB_ORG."},
)

git_local_repo_root: str = field(
    default=None,
    metadata={
        "description": "Filesystem root that contains local repos (e.g. /home/user/gh). Env GIT_LOCAL_REPO_ROOT."
    },
)

""" </Data Analysis Agent — Git/GitHub> """

""" <Data Analysis Agent — Cursor> """

cursor_api_key: str = field(
    default=None,
    metadata={
        "description": "Cursor cloud agents API key (sk-cursor-...). Env CURSOR_API_KEY."
    },
)

cursor_local_user_dir: str = field(
    default=None,
    metadata={
        "description": "Path to ~/.config/Cursor/User used to harvest local edit history. Env CURSOR_LOCAL_USER_DIR."
    },
)

""" </Data Analysis Agent — Cursor> """

""" <Data Analysis Agent — ChatGPT/OpenAI usage> """

openai_admin_api_key: str = field(
    default=None,
    metadata={
        "description": "OpenAI Admin API key (sk-admin-...) used to call /v1/organization/usage and /costs. Env OPENAI_ADMIN_API_KEY."
    },
)

openai_org_id: str = field(
    default=None,
    metadata={"description": "OpenAI organization id (org-...). Env OPENAI_ORG_ID."},
)

""" </Data Analysis Agent — ChatGPT/OpenAI usage> """

""" <Data Analysis Agent — Azure> """

azure_tenant_id: str = field(
    default=None,
    metadata={"description": "Azure AD tenant id. Env AZURE_TENANT_ID."},
)

azure_client_id: str = field(
    default=None,
    metadata={"description": "Azure AD app/client id. Env AZURE_CLIENT_ID."},
)

azure_client_secret: str = field(
    default=None,
    metadata={"description": "Azure AD app client secret. Env AZURE_CLIENT_SECRET."},
)

azure_subscription_id: str = field(
    default=None,
    metadata={
        "description": "Azure subscription id used as default Cost Management scope. Env AZURE_SUBSCRIPTION_ID."
    },
)

""" </Data Analysis Agent — Azure> """

""" <Data Analysis Agent — AWS> """

aws_access_key_id: str = field(
    default=None,
    metadata={"description": "AWS access key id. Env AWS_ACCESS_KEY_ID."},
)

aws_secret_access_key: str = field(
    default=None,
    metadata={"description": "AWS secret access key. Env AWS_SECRET_ACCESS_KEY."},
)

aws_default_region: str = field(
    default="us-east-1",
    metadata={
        "description": "Default AWS region. Cost Explorer requires us-east-1. Env AWS_DEFAULT_REGION."
    },
)

aws_profile: str = field(
    default=None,
    metadata={
        "description": "Optional AWS profile name. When set, boto3.Session(profile_name=...) is used. Env AWS_PROFILE."
    },
)

""" </Data Analysis Agent — AWS> """
```

## .env / .env.dev keys to add (uppercase, mirroring above)

```bash
# Sandbox
SANDBOX_PROVIDER=local
SANDBOX_ROOT_DIR=/tmp/anubis_sandbox
DAYTONA_API_KEY=
MODAL_TOKEN_ID=
MODAL_TOKEN_SECRET=
RUNLOOP_API_KEY=

# Slack
SLACK_BOT_TOKEN=
SLACK_USER_TOKEN=
SLACK_SIGNING_SECRET=
SLACK_APP_TOKEN=
SLACK_DEFAULT_CHANNEL=
SLACK_HISTORY_MAX_PAGES=10

# Sleep
SLEEP_PROVIDER=apple
OURA_API_TOKEN=
WHOOP_CLIENT_ID=
WHOOP_CLIENT_SECRET=
FITBIT_CLIENT_ID=
FITBIT_CLIENT_SECRET=

# BoA via Plaid
PLAID_CLIENT_ID=
PLAID_SECRET=
PLAID_ENV=sandbox
PLAID_BOA_ACCESS_TOKEN=

# Google
GOOGLE_OAUTH_CLIENT_SECRET_JSON=
GOOGLE_TOKEN_JSON_PATH=
GOOGLE_MEET_SCOPES=https://www.googleapis.com/auth/meetings.space.readonly,https://www.googleapis.com/auth/drive.readonly,https://www.googleapis.com/auth/documents.readonly

# Git / GitHub
GITHUB_PAT=
GITHUB_ORG=
GIT_LOCAL_REPO_ROOT=/home/user/gh

# Cursor
CURSOR_API_KEY=
CURSOR_LOCAL_USER_DIR=/home/user/.config/Cursor/User

# ChatGPT usage
OPENAI_ADMIN_API_KEY=
OPENAI_ORG_ID=

# Azure
AZURE_TENANT_ID=
AZURE_CLIENT_ID=
AZURE_CLIENT_SECRET=
AZURE_SUBSCRIPTION_ID=

# AWS
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_DEFAULT_REGION=us-east-1
AWS_PROFILE=
```

## Usage pattern (workspace rule restated)

Every connector function declared in `src/anubis/utils/tools/<source>/...` must follow:

```python
from src.anubis.utils.context import GlobalContext

@tool
def ingest_aws_cost(days: int = 30) -> str:
    """..."""
    context = GlobalContext()
    aws_key = context.aws_access_key_id
    aws_secret = context.aws_secret_access_key
    region = context.aws_default_region
    ...
```

The context is **instantiated at the top of the function** and passed down to nested helpers (i.e. do not import os.environ inside helpers — inject `context` as a parameter), exactly as `init_model` already does in `src/anubis/utils/model.py`.
