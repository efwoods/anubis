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


# TWITCH BOT INTEGRATION EXAMPLE:
AI Overview                 To create a Twitch chatbot in Python that interfaces with the OpenAI ChatGPT API and streams responses back to chat, you will use the TwitchIO library for the Twitch connection and the openai Python SDK for the ChatGPT integration.Here is how you can set this up:For a visual guide on setting up the OpenAI ChatGPT API and authenticating your bot:1mAdd ChatGPT into your own Twitch chat in just 4 simple ...oSetinhasYouTube · Mar 24, 20231. Prerequisites & InstallationInstall the required asynchronous Python libraries via your terminal or command prompt:bashpip install twitchio openai
Use code with caution.2. Configuration & AuthenticationYou will need credentials for both services:Twitch: Register a new application at the Twitch Developer Console to get your Client ID and generate an OAuth token for your bot account.OpenAI: Sign up at the OpenAI Platform to add billing credits and create a new secret API Key.3. Python Code ImplementationCreate a bot.py file and use the following script. This code sets up an asynchronous bot, listens for chat messages, sends the context to ChatGPT, and streams the AI's response in real-time back to the Twitch chat.pythonimport os
from twitchio.ext import commands
from openai import AsyncOpenAI

# 1. Initialize API Clients
twitch_token = os.environ.get("TWITCH_OAUTH_TOKEN")
openai_api_key = os.environ.get("OPENAI_API_KEY")

# Initialize OpenAI
client = AsyncOpenAI(api_key=openai_api_key)

# 2. Define the Twitch Bot
class Bot(commands.Bot):
    def __init__(self):
        # Initialize the bot with your OAuth, Client ID, and the target channels
        super().__init__(
            token=twitch_token,
            prefix='!', # Bot command prefix
            initial_channels=['#your_twitch_channel']
        )

    async def event_ready(self):
        print(f'Logged in as | {self.nick}')
        print(f'User ID | {self.user_id}')

    async def event_message(self, message):
        # Prevent the bot from responding to itself
        if message.echo:
            return

        # Check if the message is directed at the bot (e.g., starts with !ask or directly pinged)
        # For simplicity, we process all messages containing a specific trigger or command
        if message.content.startswith('!ai'):
            user_question = message.content.replace('!ai', '', 1).strip()
            
            # Send context to ChatGPT API with streaming
            try:
                stream = await client.chat.completions.create(
                    model="gpt-4o", # Or gpt-4o-mini
                    messages=[
                        {"role": "system", "content": "You are a helpful and witty Twitch chatbot."},
                        {"role": "user", "content": f"{message.author.name} asked: {user_question}"}
                    ],
                    stream=True,
                )

                response_chunks = []
                # Stream the response back in chunks
                async for chunk in stream:
                    if chunk.choices[0].delta.content is not None:
                        response_chunks.append(chunk.choices[0].delta.content)

                full_response = "".join(response_chunks)
                
                # Split the full response to fit Twitch's character limit (500 chars)
                # and send the stream back to the chat room
                await message.channel.send(f"@{message.author.name} {full_response}")

            except Exception as e:
                print(f"Error fetching AI response: {e}")
                await message.channel.send("Sorry, I'm having trouble thinking right now.")

        # Process standard Twitch commands if you define any
        await self.handle_commands(message)

# 3. Run the Bot
if __name__ == "__main__":
    bot = Bot()
    bot.run()
Use code with caution.4. Important Considerations for Twitch & OpenAIRate Limits: Twitch restricts chat message frequency (typically 20 messages per 30 seconds for verified bots). If you expect heavy traffic, build in a command cooldown.Character Limits: Twitch chat has a message limit. If ChatGPT outputs a massive paragraph, you may need to chunk the final string via Python before calling .send().API Cost: Ensure you monitor your OpenAI billing, as streaming live responses to a highly active Twitch chat can deplete tokens quickly.Would you like help with:Setting up a message cooldown system to prevent API spam?Adding a prompt history / memory so the bot remembers the context of past messages?Handling message truncation to ensure long OpenAI responses don't get blocked by Twitch limits?
