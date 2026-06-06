# build a twitch bot that moderates chat using openai api to stream chat


To build an AI-powered Twitch moderation bot using Python, you need to combine the Twitch IRC protocol (or Twitch's EventSub WebSocket) with the OpenAI Moderation API. The OpenAI Moderation API is ideal for this task because it is specifically fine-tuned to detect hate speech, harassment, self-harm, and sexual content, and it is completely free to use.1. Prerequisites and API KeysBefore writing code, you need to gather your access credentials:OpenAI API Key: Create an account on the ⁠OpenAI Platform and generate a secret key.Twitch Account: Create a separate Twitch account for your bot, or use your own channel's account.Twitch OAuth Token: Log into the bot account and visit TwitchChatOAuth to generate an IRC OAuth token. It will look like oauth:xxxxxx.Mod Status: Make sure your bot is a moderator in your chat by typing /mod <bot_username> in your Twitch chat stream.Run the following command in your terminal to install the necessary libraries:bashpip install tmi.py openai
Use code with caution.2. Complete Python ImplementationSave the following code as moderator_bot.py. This script connects to your Twitch channel chat, listens to incoming messages, sends them to OpenAI's evaluation endpoint, and issues an automatic timeout or ban if the message violates safety thresholds.pythonimport os
from twitchio.ext import commands
from openai import OpenAI

# Initialize the OpenAI client (looks for OPENAI_API_KEY environment variable)
openai_client = OpenAI(api_key="YOUR_OPENAI_API_KEY")

# Twitch Bot Configuration
TWITCH_TOKEN = "oauth:YOUR_TWITCH_OAUTH_TOKEN"  # Generated from ://twitchapps.com
BOT_PREFIX = "!"
INITIAL_CHANNELS = ["YOUR_CHANNEL_NAME"]  # The channel chat you want to moderate

class ModBot(commands.Bot):

    def __init__(self):
        super().__init__(
            token=TWITCH_TOKEN, 
            prefix=BOT_PREFIX, 
            initial_channels=INITIAL_CHANNELS
        )

    async def event_ready(self):
        """Triggered when the bot successfully connects to Twitch."""
        print(f"Logged in as | {self.nick}")
        print(f"User id is   | {self.user_id}")
        print(f"Moderating   | {INITIAL_CHANNELS}")

    async def event_message(self, message):
        """Triggered every time a message is sent in the Twitch chat."""
        # Prevent the bot from analyzing its own messages
        if message.echo:
            return

        user = message.author.name
        content = message.content

        print(f"[{user}]: {content}")

        # Send text to OpenAI Moderation endpoint
        try:
            moderation_response = openai_client.moderations.create(input=content)
            result = moderation_response.results[0]

            # Check if OpenAI flagged the message
            if result.flagged:
                # Find which specific categories were triggered
                triggered_categories = [cat for cat, val in result.categories.__dict__.items() if val]
                reason = ", ".join(triggered_categories)
                
                print(f"⚠️ FLAGGED MESSAGE BY {user}: {content} (Reason: {reason})")
                
                # Execute moderation action (Timeout user for 600 seconds)
                ctx = await self.get_context(message)
                await ctx.send(f"/timeout {user} 600 AI_Flagged: {reason}")
                await ctx.send(f"🤖 Removed message from @{user} for violating chat guidelines.")
                return

        except Exception as e:
            print(f"Error checking moderation API: {e}")

        # Allows standard command processing if you add custom commands later
        await self.handle_commands(message)

# Run the bot
if __name__ == "__main__":
    bot = ModBot()
    bot.run()
Use code with caution.3. Execution InstructionsTo execute and host your bot locally, initialize your API variables directly in your command line terminal interface:bash# On Linux/macOS
export OPENAI_API_KEY="your-openai-key"
python moderator_bot.py

# On Windows (Command Prompt)
set OPENAI_API_KEY="your-openai-key"
python moderator_bot.py
Use code with caution.4. Code Architecture OverviewTwitchIO framework: The bot inherits from commands.Bot to establish an asynchronous architecture. It connects via an IRC WebSocket wrapper to stream the chat data in real-time.event_message hook: Every payload arriving from Twitch chat goes through this asynchronous function wrapper.OpenAI moderations.create: The bot makes a synchronous HTTP request over the network to OpenAI's server infrastructure. It evaluates the text for harassment, hate, self-harm, sexual content, and violence./timeout action command: If the payload parameter result.flagged resolves to True, the bot sends an active moderation text string back into the channel room chat interface. This triggers a server-side client timeout for the targeted violator.If you'd like to extend this setup, let me know if you want the bot to respond with customized AI warnings instead of an immediate timeout, or if you need help deploying it to run 24/7 using a free hosting platform.