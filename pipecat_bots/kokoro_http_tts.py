"""HTTP client for Streaming Kokoro TTS server.

Connects to the Kokoro TTS HTTP server for speech synthesis.
Runs on the host - no NeMo/PyTorch dependencies required.

Usage:
    tts = KokoroHTTPTTSService(server_url="http://localhost:8001")
    # In pipeline: ... -> llm -> tts -> transport.output() -> ...
"""

import asyncio
from typing import AsyncGenerator, Optional

import httpx
import requests
from loguru import logger
from pydantic import BaseModel

from pipecat.frames.frames import (
    ErrorFrame,
    EndFrame,
    LLMTextFrame,
    LLMFullResponseEndFrame,
    Frame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)

from pipecat.services.tts_service import TTSService
from pipecat.processors.frame_processor import FrameDirection

# Import the continue frame for LLM/TTS synchronization
# Use try/except to support both local execution and Modal deployment
try:
    from pipecat_bots.frames import ChunkedLLMContinueGenerationFrame
except ModuleNotFoundError:
    from frames import ChunkedLLMContinueGenerationFrame

try:
    from websockets.asyncio.client import connect as websocket_connect
    from websockets.protocol import State
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("Install websockets: pip install websockets")
    raise


# Default sample rate (will be fetched from server)
DEFAULT_SAMPLE_RATE = 22000


class KokoroHTTPTTSService(TTSService):
    """HTTP client for Kokoro TTS server.

    Connects to a Kokoro Streaming TTS HTTP server for speech synthesis.
    No NeMo/PyTorch dependencies - runs on host.
    """

    class InputParams(BaseModel):
        """Input parameters for HTTP TTS."""

        language: str = "en"

    def __init__(
        self,
        *,
        server_url: str = "http://localhost:8001",
        voice: str = "af_heart",
        language: str = "en",
        sample_rate: Optional[int] = None,
        params: Optional[InputParams] = None,
        **kwargs,
    ):
        """Initialize Kokoro HTTP TTS client.

        Args:
            server_url: TTS server URL (default: http://localhost:8001)
            voice: Speaker voice (af_heart, af_bella, see kokoro-fastapi project for more)
            language: Language code (en, ...)
            sample_rate: Output sample rate (default: fetched from server)
            params: Additional TTS parameters.
        """
        # Will update sample_rate after fetching from server
        super().__init__(sample_rate=sample_rate or DEFAULT_SAMPLE_RATE, **kwargs)

        params = params or KokoroHTTPTTSService.InputParams()

        self._server_url = server_url.rstrip("/")
        self._voice = voice.lower()
        self._language = language.lower()
        self._sample_rate = sample_rate

        # HTTP client with connection pooling
        self._client = httpx.AsyncClient(timeout=30.0)
        self._config_fetched = False

        self.set_model_name("kokoro-http")
        self.set_voice(voice)

        logger.info(
            f"KokoroHTTPTTS initialized: server={server_url}, "
            f"voice={voice}, language={language}"
        )

    async def _ensure_config(self):
        """Fetch server config if not already done."""
        if self._config_fetched:
            return

        try:
            server_sample_rate = DEFAULT_SAMPLE_RATE
            self._config_fetched = True
        except Exception as e:
            logger.warning(f"Failed to fetch TTS config: {e}")
            self._config_fetched = True  # Don't retry on every request

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics."""
        return True

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames with special handling for LLM response end."""
        await super().process_frame(frame, direction)

        if isinstance (frame, (LLMTextFrame)):
            # Signal ChunkedLLMService that this segment is complete
            # so it can continue generating the next chunk
            # This will kickoff run_tts initially
            await self.push_frame(
                ChunkedLLMContinueGenerationFrame(),
                FrameDirection.UPSTREAM
            )

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using Kokoro TTS HTTP server.

        Args:
            text: The text to synthesize.

        Yields:
            TTSStartedFrame, TTSAudioRawFrame, TTSStoppedFrame
        """
        await self.start_ttfb_metrics()
        yield TTSStartedFrame()

        # Fetch config on first request
        await self._ensure_config()

        # Normalize unicode characters
        text = text.replace("\u2018", "'")  # LEFT SINGLE QUOTATION MARK
        text = text.replace("\u2019", "'")  # RIGHT SINGLE QUOTATION MARK
        text = text.replace("\u201C", '"')  # LEFT DOUBLE QUOTATION MARK
        text = text.replace("\u201D", '"')  # RIGHT DOUBLE QUOTATION MARK
        text = text.replace("\u2014", "-")  # EM DASH
        text = text.replace("\u2013", "-")  # EN DASH

        logger.debug(f"KokoroHTTPTTS: Generating [{text[:50]}...]")

        try:
            async with self._client.stream(
                "POST",
                f"{self._server_url}/v1/audio/speech",
                json={
                    "input": text,
                    "voice": self._voice,
                    #"language": self._language,
                    "response_format": "pcm",
                    }) as resp:

                if resp.status_code != 200:
                    error_msg = f"TTS server error: {resp.status_code} - {resp.text}"
                    logger.error(error_msg)
                    yield ErrorFrame(error=error_msg)
                    yield TTSStoppedFrame()
                    return

                await self.stop_ttfb_metrics()
                async for chunk in resp.aiter_bytes(chunk_size=1024):
                    if chunk:
                        audio_bytes = chunk

                        yield TTSAudioRawFrame(
                            audio=audio_bytes,
                            sample_rate=DEFAULT_SAMPLE_RATE,
                            num_channels=1,
                        )
            await self.push_frame(
                ChunkedLLMContinueGenerationFrame(),
                FrameDirection.UPSTREAM
            )
            await self.start_tts_usage_metrics(text)
            yield TTSStoppedFrame()

        except httpx.ConnectError as e:
            error_msg = f"Cannot connect to TTS server at {self._server_url}: {e}"
            logger.error(error_msg)
            yield ErrorFrame(error=error_msg)
            yield TTSStoppedFrame()

        except Exception as e:
            logger.error(f"KokoroHTTPTTS error: {e}")
            yield ErrorFrame(error=str(e))
            yield TTSStoppedFrame()

    async def close(self):
        """Close HTTP client."""
        await self._client.aclose()

    def set_voice(self, voice: str):
        """Change the speaker voice.

        Args:
            voice: Speaker name (john, sofia, aria, jason, leo).
        """
        self._voice = voice.lower()
        super().set_voice(voice)
        logger.info(f"KokoroHTTPTTS: Voice changed to {voice}")

    def set_language(self, language: str):
        """Change the language.

        Args:
            language: Language code (en, es, de, fr, vi, it, zh).
        """
        self._language = language.lower()
        logger.info(f"KokoroHTTPTTS: Language changed to {language}")
