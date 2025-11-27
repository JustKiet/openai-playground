from __future__ import annotations
import asyncio
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, AsyncGenerator
from langchain_openai import ChatOpenAI

# -----------------------------
# EventBus (Observer Pattern)
# -----------------------------
class EventBus:
    def __init__(self) -> None:
        self.listeners: Dict[str, List[Callable[..., Any]]] = defaultdict(list)

    def subscribe(self, event_name: str, callback: Callable[..., Any]) -> None:
        self.listeners[event_name].append(callback)

    async def emit(self, event_name: str, *args: Any, **kwargs: Any) -> None:
        if event_name in self.listeners:
            await asyncio.gather(
                *[listener(*args, event_name=event_name, **kwargs) for listener in self.listeners[event_name]]
            )

# -----------------------------
# LLM Stub
# -----------------------------
class LLM:
    def __init__(self, bus: Optional[EventBus] = None) -> None:
        self.bus = bus

    async def run_with_events(self, prompt: str) -> None:
        print(f"[LLM] Received prompt: {prompt}")

        # Simulate reasoning: decides it needs a tool
        await asyncio.sleep(0.2)
        tool = ("get_weather", {"city": "Tokyo"})
        print("[LLM] Emitting tool_call event:", tool)
        if self.bus:
            await self.bus.emit("llm.tool_call", *tool)
        else:
            raise RuntimeError("Event mode requires EventBus!")

    async def continue_with_tool_result(self, result: Dict[str, Any]) -> None:
        print("[LLM] Received tool result:", result)
        await asyncio.sleep(0.2)
        final_output = f"The weather in {result['city']} is {result['temp']}°C."
        print("[LLM] Emitting final_output event")
        if self.bus:
            await self.bus.emit("llm.final_output", final_output)

# -----------------------------
# ToolHandler Stub (formerly Agent)
# -----------------------------
class ToolHandler:
    def __init__(self, bus: EventBus) -> None:
        self.bus = bus

    async def on_tool_call(self, tool_name: str, tool_args: Dict[str, Any], **kwargs: Any) -> None:
        print(f"[TOOL_HANDLER] Handling tool call: {tool_name}({tool_args})")
        await asyncio.sleep(0.3)  # simulate work
        # Simulate tool execution
        result: Dict[str, Any] = {}
        if tool_name == "get_weather":
            result = {"city": tool_args["city"], "temp": 22}
        print("[TOOL_HANDLER] Emitting tool_result event:", result)
        await self.bus.emit("agent.tool_result", result)

# -----------------------------
# State Pattern
# -----------------------------
class BaseState:
    async def on_event(self, session: Session, event_name: str, *args: Any, **kwargs: Any) -> None:
        pass

class Idle(BaseState):
    async def on_event(self, session: Session, event_name: str, *args: Any, **kwargs: Any) -> None:
        if event_name == "start":
            session.state = LLMThinking()
            await session.llm.run_with_events(*args)

class LLMThinking(BaseState):
    async def on_event(self, session: Session, event_name: str, *args: Any, **kwargs: Any) -> None:
        if event_name == "llm.tool_call":
            session.state = WaitingForTool()
        elif event_name == "llm.final_output":
            session.state = Finished()

class WaitingForTool(BaseState):
    async def on_event(self, session: Session, event_name: str, *args: Any, **kwargs: Any) -> None:
        if event_name == "agent.tool_result":
            session.state = LLMThinking()
            await session.llm.continue_with_tool_result(*args)

class Finished(BaseState):
    async def on_event(self, session: Session, event_name: str, *args: Any, **kwargs: Any) -> None:
        print("[SESSION] Workflow completed.")

# -----------------------------
# Session (State Machine + Event Router)
# -----------------------------
class Session:
    def __init__(self, bus: EventBus, llm: LLM) -> None:
        self.bus = bus
        self.llm = llm
        self.state: BaseState = Idle()

        # Subscribe to all relevant events
        for event_name in ["start", "llm.tool_call", "agent.tool_result", "llm.final_output"]:
            bus.subscribe(event_name, self._on_event)

    async def _on_event(self, *args: Any, event_name: Optional[str] = None, **kwargs: Any) -> None:
        if event_name:
            await self.state.on_event(self, event_name, *args, **kwargs)

    async def start(self, prompt: str) -> None:
        await self.bus.emit("start", prompt)

# -----------------------------
# Agent (Main Controller)
# -----------------------------
class Agent:
    def __init__(self) -> None:
        self.bus = EventBus()
        self.llm = LLM(self.bus)
        self.tool_handler = ToolHandler(self.bus)
        self.session = Session(self.bus, self.llm)

        # Subscribe tool handler to tool calls
        self.bus.subscribe("llm.tool_call", self.tool_handler.on_tool_call)

    async def run(self, prompt: str) -> AsyncGenerator[Any, None]:
        queue: asyncio.Queue[Any] = asyncio.Queue()

        async def event_listener(*args: Any, event_name: str, **kwargs: Any) -> None:
            await queue.put((event_name, args, kwargs))

        # Subscribe to all relevant events
        for event_name in ["start", "llm.tool_call", "agent.tool_result", "llm.final_output"]:
            self.bus.subscribe(event_name, event_listener)

        task = asyncio.create_task(self.session.start(prompt))

        while not task.done():
            # Wait for either a new event or the task to finish
            get_task = asyncio.create_task(queue.get())
            done, _ = await asyncio.wait({task, get_task}, return_when=asyncio.FIRST_COMPLETED)

            if get_task in done:
                yield get_task.result()
            else:
                # Task finished, cancel the get_task
                get_task.cancel()
        
        # Drain any remaining events
        while not queue.empty():
            yield queue.get_nowait()

# -----------------------------
# Run everything
# -----------------------------
async def main() -> None:
    agent = Agent()
    async for event in agent.run("What's the weather in Tokyo?"):
        print(f"[MAIN] Event received: {event}")

if __name__ == "__main__":
    asyncio.run(main())