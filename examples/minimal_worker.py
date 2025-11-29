import logging
from dotenv import load_dotenv

from livekit.agents import AgentServer, JobContext, cli
from livekit.agents.stt import SpeechEvent

from livekit.agents.vad import VoiceActivity
from livekit.agents.tts import TTS

# Import your Interrupt Handler
from livekit.agents.interrupt_handler import InterruptConfig, InterruptHandler

logger = logging.getLogger("minimal-worker")
logger.setLevel(logging.INFO)

load_dotenv()

server = AgentServer()

# ---------------------------
# SETUP INTERRUPT HANDLER
# ---------------------------
cfg = InterruptConfig(
    validation_timeout_ms=150,
    ignore_list=["yeah", "ok", "hmm", "uh-huh", "right", "mm"],
    interrupt_list=["stop", "wait", "no", "pause"]
)
ih = InterruptHandler(cfg)


# Callback: When user truly interrupts
def on_interrupt(transcript: str):
    logger.info(f"INTERRUPT VALIDATED => {transcript}")
    # stop TTS immediately
    if ctx.tts_handle:
        ctx.tts_handle.interrupt(force=True)


# Callback: Ignore filler/backchannel
def on_ignore(transcript: str, reason: str):
    logger.info(f"IGNORED ({reason}) => {transcript}")
    # do nothing, continue speaking


# Callback: Agent silent => treat as normal input
def on_respond(transcript: str):
    logger.info(f"RESPOND => {transcript}")
    # You may send this transcript to LLM here if needed


# Assign callbacks
ih.on_validated_interrupt = on_interrupt
ih.on_ignore = on_ignore
ih.on_respond = on_respond


# ---------------------------
# MAIN ENTRYPOINT
# ---------------------------
@server.rtc_session()
async def entrypoint(ctx: JobContext):

    logger.info(f"Connected to room {ctx.room.name}")

    # Listen to TTS start/stop
    @ctx.on_tts_start
    async def _on_tts_start(handle: TTS):
        ih.set_speaking(True)

    @ctx.on_tts_end
    async def _on_tts_end(handle: TTS):
        ih.set_speaking(False)

    # Listen for VAD events
    @ctx.on_vad
    async def _on_vad(event: VoiceActivity):
        if event.is_speech:
            ih.handle_vad_start()

    # Listen for STT events (partial/final)
    @ctx.on_stt
    async def _on_stt(event: SpeechEvent):
        if event.is_partial:
            ih.handle_partial_transcript(event.text)
        elif event.is_final:
            ih.handle_final_transcript(event.text)

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(server)
