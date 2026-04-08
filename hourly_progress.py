import subprocess
import httpx
import os
import json
import argparse

from dotenv import load_dotenv
load_dotenv()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hourly Progress Script")
    # parser.add_argument("--repo_abs_path", nargs="?", help="Optional path to root of the git directory. Otherwise the cwd is used for git commit history.")
    parser.add_argument("--NN_API_KEY", nargs="?", help="Optional path to input file")
    parser.add_argument("--current", help="Logs current updates since the last commit", action="store_true")
    args = parser.parse_args()

    API_KEY = os.environ.get("NN_API_KEY")

    if not API_KEY and not args.NN_API_KEY:
        # input("Enter Neural Nexus API key:")
        raise Exception("Please enter your Neural Nexus API key as NN_API_KEY in your environment.")
    
    if not API_KEY:
        API_KEY = args.NN_API_KEY

    # if not args.repo_abs_path:
        # print("filepath not set: using current working directory.")
    progress_file_path = os.getcwd() + "/progress_1_hour.txt"
    # else:
        # progress_file_path = args.repo_abs_path + "/progress_1_hour.txt"

    print(f"{progress_file_path}")
    # if args.repo_abs_path:
        # subprocess.run(f"cd {args.repo_abs_path}")
    subprocess.run(f"git diff HEAD > progress_1_hour.txt", shell=True, text=True)
    if not args.current:
        print("Reviewing hourly progress...")
        system_message = "SPEAK USING YOUR PERSONALITY AND PERSONAL EXPERIENCE AS IF YOU MADE THESE CHANGES. YOU MADE THESE CHANGES. BE CLEAR, SUCCINCT, AND ACCURATE IN YOUR RESPONSES. ONLY REFER TO THE FOLLOWING INSTRUCTIONS.  Please describe what has been changed within the last hour from the following text and use your own style of writing. Write as if you made the code changes yourself:"
        message = "Please describe what has been changed within the last hour from the following text:"
    else:
        print("Reviewing progress...")
        system_message = "SPEAK USING YOUR PERSONALITY AND PERSONAL EXPERIENCE AS IF YOU MADE THESE CHANGES. YOU MADE THESE CHANGES. BE CLEAR, SUCCINCT, AND ACCURATE IN YOUR RESPONSES. ONLY REFER TO THE FOLLOWING INSTRUCTIONS.  Please describe what has been changed since the last commit from the following text and use your own style of writing. Write as if you made the code changes yourself:"
        message = "Please describe what has been changed since the last commit from the following text:"
    with open(progress_file_path, 'rb') as fp:
        response = httpx.post(url="https://api.neuralnexus.site/chat",
            headers={
              "API-KEY": API_KEY
            },
            data={
              "message": message,
            },
            files={"file":fp},
            timeout=httpx.Timeout(120) # timeout in seconds
        )
        update_response = json.loads(response.content.decode('utf-8')).get("content")
        fp.close()
    subprocess.run([f"git commit --allow-empty -m \"{update_response}\" && git push"], text=True, shell=True)
    subprocess.run([f"rm {progress_file_path}"], text=True, shell=True)
    print(f"{update_response}")
    
