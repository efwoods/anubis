# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "httpx",
#     "python-dotenv",
# ]
# ///
import subprocess
import httpx
import os
import json
import argparse

from dotenv import load_dotenv
load_dotenv()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hourly Progress Script")
    parser.add_argument("--NN_API_KEY", "-k", dest="nn_api_key", nargs="?", help="API Key to the Neural Nexus")
    parser.add_argument("--AVATAR_ID", "-a", dest="avatar_id", nargs="?", help="avatar_id assigned to the avatar from the Neural Nexus.")
    parser.add_argument("--current", "-c", help="Logs current updates since the last commit", action="store_true")
    parser.add_argument("--mass_commit", help="non-empty commit message including all files and changes", action="store_true")
    args = parser.parse_args()

    API_KEY = os.environ.get("NN_API_KEY") or args.nn_api_key
    if not API_KEY:
        raise Exception("Please enter your Neural Nexus API key as NN_API_KEY in your environment or pass --NN_API_KEY/-k.")

    avatar_id = os.environ.get("AVATAR_ID") or args.avatar_id
    if not avatar_id:
        raise Exception("Please select an AVATAR using the assistant id. Set AVATAR_ID in your environment or pass --AVATAR_ID/-a. Use --help for usage.")

    progress_file_path = os.getcwd() + "/progress_1_hour.txt"
    worktree_name = os.path.basename(os.getcwd())

    print(f"{progress_file_path}")
    subprocess.run(f"git diff HEAD > progress_1_hour.txt", shell=True, text=True)
    if not args.current:
        print("Reviewing hourly progress...")
        system_message = f"<INSTRUCTIONS>SPEAK USING YOUR PERSONALITY AND PERSONAL EXPERIENCE AS IF YOU MADE THESE CHANGES. YOU MADE THESE CHANGES. BE CLEAR, SUCCINCT, AND ACCURATE IN YOUR RESPONSES. ONLY REFER TO THE FOLLOWING INSTRUCTIONS. You are working in the '{worktree_name}' worktree; reference the '{worktree_name}' worktree when describing your work. Please describe what has been changed within the last hour from the following text and use your own style of writing. Write as if you made the code changes yourself and only reference any work that was actually listed as a change in the git diff. If there are no changes, indicate that there are no changes. Do not ask for follow up information as you are immediately posting an update. Do not talk about anything other than then changes to the code.  Do not include information that was not listed in the git diff:</INSTRUCTIONS>Please describe what has been changed since the last commit from the following text:"
    else:
        print("Reviewing progress...")
        system_message = f"""<INSTRUCTIONS>SPEAK USING YOUR PERSONALITY AND PERSONAL EXPERIENCE AS IF YOU MADE THESE CHANGES. YOU MADE THESE CHANGES. BE CLEAR, SUCCINCT, AND ACCURATE IN YOUR RESPONSES. ONLY REFER TO THE FOLLOWING INSTRUCTIONS. You are working in the '{worktree_name}' worktree; reference the '{worktree_name}' worktree when describing your work. Please describe what has been changed since the last commit from the following text and use your own style of writing. Write as if you made the code changes yourself and only reference any work that was actually listed as a change in the git diff. If there are no changes, indicate that there are no changes. Do not ask for follow up information as you are immediately posting an update. Do not talk about anything other than then changes to the code. Do not include information that was not listed in the git diff:</INSTRUCTIONS>Please describe what has been changed since the last commit from the following text:"""
    with open(progress_file_path, 'rb') as fp:
        response = httpx.post(url=f"https://api.neuralnexus.site/message/{avatar_id}",
            headers={
              "API-KEY": API_KEY
            },
            data={
              "message": system_message,
              "stream": "false",
            },
            files={"files": ("progress_1_hour.txt", fp, "text/plain")},
            timeout=None
        )
        if response.status_code != 200:
            raise Exception(
                f"Neural Nexus request failed ({response.status_code}): {response.text[:500]}"
            )
        update_response = json.loads(response.content.decode('utf-8')).get("content")
    with open(progress_file_path, "w") as fp:
        fp.write(update_response)
    # update_response_str = update_response.replace("'", "'\"'\"'")
    # UPDATE_COMMAND = "git commit --allow-empty -m '" + update_response_str + "' && git push"
    if args.mass_commit:
        UPDATE_COMMAND = "git add -A && git commit -F '" + progress_file_path + "' && git push"
    else:
        UPDATE_COMMAND = "git commit --allow-empty -F '" + progress_file_path + "' && git push"
    # print(f"{UPDATE_COMMAND}")
    # print(f"\n\n")
    subprocess.run([f"{UPDATE_COMMAND}"], text=True, shell=True)
    subprocess.run([f"rm {progress_file_path}"], text=True, shell=True)
    print(f"{update_response}")