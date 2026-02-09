"""
Discord Bot for Auto-Trader
============================
Responds to messages in #trading channel, provides:
- Market status and briefings
- Free-form LLM conversations with market context

Only responds to messages from the authorized user (DISCORD_OWNER_ID).
"""
import os
import sys
import json
import logging
import asyncio
from datetime import datetime

import discord
from discord.ext import commands
from dotenv import load_dotenv

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.llm_client_genai import GeminiClient

# Load environment
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("DiscordBot")

# Config
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
DISCORD_OWNER_ID = os.getenv("DISCORD_OWNER_ID")  # Your Discord User ID
TRADING_CHANNEL_NAME = os.getenv("DISCORD_CHANNEL_NAME", "trading")

# Data paths
MARKET_CONTEXT_FILE = "data/market_context.json"
BRIEFING_FILE = "data/latest_briefing.md"

# Intents
intents = discord.Intents.default()
intents.message_content = True  # Required to read message content

bot = commands.Bot(command_prefix="!", intents=intents)

# LLM Client
llm_client = None


def load_market_context():
    """Load the latest market context from file."""
    try:
        if os.path.exists(MARKET_CONTEXT_FILE):
            with open(MARKET_CONTEXT_FILE, 'r') as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load market context: {e}")
    return None


def load_briefing():
    """Load the latest market briefing from file."""
    try:
        if os.path.exists(BRIEFING_FILE):
            with open(BRIEFING_FILE, 'r', encoding='utf-8') as f:
                return f.read()
    except Exception as e:
        logger.error(f"Failed to load briefing: {e}")
    return None


def format_status_embed(ctx_data):
    """Format market context into a Discord embed."""
    embed = discord.Embed(
        title="📊 Market Status",
        color=0x3498DB,
        timestamp=datetime.utcnow()
    )
    
    if ctx_data:
        embed.add_field(
            name="Risk / Bias / Sentiment",
            value=f"**{ctx_data.get('risk_multiplier', 'N/A')}** / **{ctx_data.get('bias', 'N/A')}** / **{ctx_data.get('sentiment_score', 'N/A')}**",
            inline=False
        )
        embed.add_field(
            name="Regime",
            value=ctx_data.get('regime', 'unknown'),
            inline=True
        )
        embed.add_field(
            name="Last Update",
            value=ctx_data.get('timestamp', 'N/A')[:19].replace('T', ' '),
            inline=True
        )
        
        reasoning = ctx_data.get('reasoning', '')
        if reasoning:
            embed.add_field(
                name="Reasoning",
                value=reasoning[:500],
                inline=False
            )
    else:
        embed.description = "⚠️ No market context available."
    
    return embed


async def query_llm_with_context(user_message: str) -> str:
    """Query LLM with market context injected."""
    global llm_client
    
    if llm_client is None:
        llm_client = GeminiClient()
    
    # Load context
    ctx = load_market_context()
    context_str = ""
    if ctx:
        context_str = f"""
Current Market Context:
- Regime: {ctx.get('regime', 'unknown')}
- Risk Multiplier: {ctx.get('risk_multiplier', 'N/A')}
- Bias: {ctx.get('bias', 'N/A')}
- Sentiment Score: {ctx.get('sentiment_score', 'N/A')}
- Reasoning: {ctx.get('reasoning', 'N/A')}
- Last Update: {ctx.get('timestamp', 'N/A')}
"""
    
    system_prompt = f"""You are the AI assistant for an automated crypto trading bot.
You have access to real-time market analysis data.

{context_str}

Answer the user's questions based on this context. Be concise and actionable.
If the user asks about market conditions, use the context above.
If they ask about something unrelated, answer normally.
Keep responses under 300 words for Discord readability."""
    
    try:
        response = llm_client.query(user_message, system_prompt, temperature=0.7)
        return response if response else "⚠️ LLM returned empty response."
    except Exception as e:
        logger.error(f"LLM query failed: {e}")
        return f"⚠️ Error querying LLM: {str(e)[:100]}"


def is_owner(message):
    """Check if message is from the authorized owner."""
    if not DISCORD_OWNER_ID:
        logger.warning("DISCORD_OWNER_ID not set, allowing all users.")
        return True
    return str(message.author.id) == str(DISCORD_OWNER_ID)


@bot.event
async def on_ready():
    logger.info(f"✅ Bot logged in as {bot.user}")
    logger.info(f"   Listening in channel: #{TRADING_CHANNEL_NAME}")
    logger.info(f"   Owner ID: {DISCORD_OWNER_ID or 'NOT SET (all users allowed)'}")


@bot.event
async def on_message(message):
    # Ignore bot's own messages
    if message.author == bot.user:
        return
    
    # Only respond in the trading channel
    if message.channel.name != TRADING_CHANNEL_NAME:
        return
    
    # Only respond to owner
    if not is_owner(message):
        logger.info(f"Ignored message from non-owner: {message.author.id}")
        return
    
    content = message.content.strip().lower()
    
    # --- Command: Status ---
    if content in ['状态', 'status', '!status']:
        ctx_data = load_market_context()
        embed = format_status_embed(ctx_data)
        await message.channel.send(embed=embed)
        return
    
    # --- Command: Briefing ---
    if content in ['简报', 'briefing', '!briefing']:
        briefing = load_briefing()
        if briefing:
            # Discord message limit is 2000 chars
            if len(briefing) > 1900:
                briefing = briefing[:1900] + "\n\n...(truncated)"
            await message.channel.send(f"```markdown\n{briefing}\n```")
        else:
            await message.channel.send("⚠️ No briefing available.")
        return
    
    # --- Command: Help ---
    if content in ['帮助', 'help', '!help']:
        help_text = """**🤖 Auto-Trader Bot Commands**

`状态` / `status` - 查看当前市场状态
`简报` / `briefing` - 查看最新市场简报
`帮助` / `help` - 显示此帮助信息

其他任何消息 - 与 AI 对话（带市场上下文）"""
        await message.channel.send(help_text)
        return
    
    # --- Free-form LLM Query ---
    async with message.channel.typing():
        response = await asyncio.get_event_loop().run_in_executor(
            None, 
            lambda: asyncio.run(async_query_wrapper(message.content))
        )
        
        # Split long responses
        if len(response) > 1900:
            response = response[:1900] + "\n\n...(truncated)"
        
        await message.channel.send(response)


async def async_query_wrapper(user_message):
    """Wrapper to call sync LLM function."""
    return await asyncio.to_thread(query_llm_with_context_sync, user_message)


def query_llm_with_context_sync(user_message: str) -> str:
    """Synchronous version for thread execution."""
    global llm_client
    
    if llm_client is None:
        llm_client = GeminiClient()
    
    ctx = load_market_context()
    context_str = ""
    if ctx:
        context_str = f"""
Current Market Context:
- Regime: {ctx.get('regime', 'unknown')}
- Risk Multiplier: {ctx.get('risk_multiplier', 'N/A')}
- Bias: {ctx.get('bias', 'N/A')}
- Sentiment Score: {ctx.get('sentiment_score', 'N/A')}
- Reasoning: {ctx.get('reasoning', 'N/A')}
- Last Update: {ctx.get('timestamp', 'N/A')}
"""
    
    system_prompt = f"""You are the AI assistant for an automated crypto trading bot.
You have access to real-time market analysis data.

{context_str}

Answer the user's questions based on this context. Be concise and actionable.
If the user asks about market conditions, use the context above.
If they ask about something unrelated, answer normally.
Keep responses under 300 words for Discord readability.
Respond in the same language as the user's message."""
    
    try:
        response = llm_client.query(user_message, system_prompt, temperature=0.7)
        return response if response else "⚠️ LLM returned empty response."
    except Exception as e:
        logger.error(f"LLM query failed: {e}")
        return f"⚠️ Error querying LLM: {str(e)[:100]}"


def main():
    if not DISCORD_BOT_TOKEN:
        logger.error("❌ DISCORD_BOT_TOKEN not set in environment!")
        logger.error("   Please add DISCORD_BOT_TOKEN to your .env file.")
        sys.exit(1)
    
    logger.info("🚀 Starting Discord Bot...")
    bot.run(DISCORD_BOT_TOKEN)


if __name__ == "__main__":
    main()
