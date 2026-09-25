"""Interactive hidden token prompt; tokens are never command-line arguments."""
from getpass import getpass
from modelscope.hub.api import HubApi

if __name__ == "__main__":
    token = getpass("ModelScope write token (hidden): ")
    if not token.strip():
        raise SystemExit("Empty token")
    HubApi().login(token.strip())
    print("ModelScope login completed. No token has been written into this repository.")
