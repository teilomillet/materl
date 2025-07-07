# materl.functions.adapters

# This file provides adapter functions to bridge synchronous and asynchronous code,
# enabling a phased transition to a fully async-native pipeline. This is a key
# part of the framework's design, allowing for gradual migration and reuse of
# existing synchronous components.

import asyncio
from typing import Callable, Any, Coroutine


def make_async(sync_fn: Callable) -> Callable[..., Coroutine[Any, Any, Any]]:
    """
    Wraps a synchronous function to be safely called from an async context.

    This is a simple but powerful adapter that allows us to reuse our existing,
    synchronous pipeline functions (e.g., for rewards, logprobs) within the new
    asynchronous RLEngine without rewriting them immediately. It runs the
    synchronous code in a separate thread pool to avoid blocking the main
    asyncio event loop.

    Args:
        sync_fn: A regular, synchronous function to wrap.

    Returns:
        An awaitable async function that executes the synchronous code
        non-blockingly.
    """
    async def wrapper(*args, **kwargs):
        loop = asyncio.get_running_loop()
        # run_in_executor runs the sync function in a separate thread,
        # preventing it from blocking the async event loop.
        return await loop.run_in_executor(None, lambda: sync_fn(*args, **kwargs))
    
    return wrapper 