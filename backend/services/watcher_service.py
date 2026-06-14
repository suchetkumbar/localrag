"""
Folder watcher service.
Monitors the watch_folder for new files and auto-ingests them.
Uses watchdog for filesystem event detection.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Set

import structlog
from watchdog.events import FileCreatedEvent, FileMovedEvent, FileSystemEventHandler
from watchdog.observers import Observer

from config.config import Settings

logger = structlog.get_logger(__name__)


class _RagFileHandler(FileSystemEventHandler):
    """Handles file system events and queues files for ingestion."""

    def __init__(self, queue: asyncio.Queue, supported_extensions: Set[str], loop: asyncio.AbstractEventLoop):
        super().__init__()
        self._queue = queue
        self._supported = supported_extensions
        self._loop = loop
        self._seen: Set[str] = set()

    def _enqueue(self, path_str: str) -> None:
        p = Path(path_str)
        if p.suffix.lower() not in self._supported:
            return
        if path_str in self._seen:
            return
        self._seen.add(path_str)
        logger.info("watcher_file_detected", path=path_str)
        asyncio.run_coroutine_threadsafe(self._queue.put(p), self._loop)

    def on_created(self, event: FileCreatedEvent) -> None:
        if not event.is_directory:
            self._enqueue(event.src_path)

    def on_moved(self, event: FileMovedEvent) -> None:
        if not event.is_directory:
            self._enqueue(event.dest_path)


class FolderWatcherService:
    """
    Watches a directory for new documents and triggers ingestion.
    Runs the watchdog Observer in a background thread.
    The async consumer runs in the main event loop.
    """

    def __init__(self, settings: Settings, ingestion_service, db_session_factory):
        self.settings = settings
        self.ingestion_service = ingestion_service
        self.db_session_factory = db_session_factory
        self._queue: asyncio.Queue = asyncio.Queue()
        self._observer: Observer | None = None
        self._consumer_task: asyncio.Task | None = None
        self._running = False

    async def start(self) -> None:
        watch_path = self.settings.watch_folder_path
        watch_path.mkdir(parents=True, exist_ok=True)

        loop = asyncio.get_event_loop()
        handler = _RagFileHandler(
            queue=self._queue,
            supported_extensions=set(self.settings.ingestion.supported_extensions),
            loop=loop,
        )

        self._observer = Observer()
        self._observer.schedule(handler, str(watch_path), recursive=False)
        self._observer.start()
        self._running = True

        # Also scan for existing files on startup
        await self._scan_existing(watch_path)

        self._consumer_task = asyncio.create_task(self._consume())
        logger.info("folder_watcher_started", path=str(watch_path))

    async def stop(self) -> None:
        self._running = False
        if self._observer:
            self._observer.stop()
            self._observer.join()
        if self._consumer_task:
            self._consumer_task.cancel()
        logger.info("folder_watcher_stopped")

    async def _scan_existing(self, folder: Path) -> None:
        """Queue any files already present in the watch folder."""
        supported = set(self.settings.ingestion.supported_extensions)
        for file_path in folder.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in supported:
                await self._queue.put(file_path)
                logger.debug("watcher_existing_file_queued", path=str(file_path))

    async def _consume(self) -> None:
        """Drain the queue and ingest each file."""
        while self._running:
            try:
                file_path: Path = await asyncio.wait_for(self._queue.get(), timeout=2.0)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            await self._safe_ingest(file_path)
            self._queue.task_done()

    async def _safe_ingest(self, file_path: Path) -> None:
        """Ingest a file, catching all errors so the consumer keeps running."""
        log = logger.bind(path=str(file_path))
        try:
            async with self.db_session_factory() as db:
                await self.ingestion_service.ingest_file(
                    file_path, db_session=db, source="watcher"
                )
                await db.commit()
            log.info("watcher_ingestion_success")
        except Exception as exc:
            log.error("watcher_ingestion_error", error=str(exc), exc_info=True)
