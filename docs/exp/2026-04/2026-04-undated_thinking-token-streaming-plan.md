# Thinking Token Streaming — Implementation Plan

**Date:** 2026-04-09
**Rollback point:** `71bf51a` on `dev` branch
**Status:** Ready for implementation

---

## Problem

During Qwen 3.6-Plus's 30–50s TTFT (thinking mode), users see only a 3-dot animation with no feedback. The thinking tokens exist in the model output but are never extracted or displayed. This feature streams thinking tokens in real-time via a collapsible "Thought process" panel, making the wait feel productive instead of broken.

---

## Phase 1: Backend — StreamDelta + Provider Extraction

### 1.1 Add `thinking_content` to StreamDelta

**File:** `src/services/assistant/models/model_registry.py:85-92`

Add field to the dataclass:

```python
thinking_content: str | None = None  # Qwen reasoning_content / Gemini thought parts
```

### 1.2 Extract `reasoning_content` in `_stream_openai()`

**File:** `src/services/assistant/models/model_registry.py:1037-1042`

Qwen 3.x via DashScope compatible-mode emits `reasoning_content` in the delta alongside `content`. Add extraction:

```python
yield StreamDelta(
    content=delta.get("content", ""),
    tool_calls=delta.get("tool_calls"),
    finish_reason=finish_reason,
    usage=usage_data,
    thinking_content=delta.get("reasoning_content"),  # NEW
)
```

### 1.3 Enable thinking for DashScope in `_build_openai_body()`

**File:** `src/services/assistant/models/model_registry.py:564-614`

- Add `thinking_level: str | None = None` parameter
- When `thinking_level` and model is `qwen3*`, add `"extra_body": {"enable_thinking": True}`

**File:** `src/services/assistant/models/model_registry.py:559-562` (`_build_request_body`)

- Pass `thinking_level` to `_build_openai_body()` (currently only passed to Google)

### 1.4 Extract Gemini thinking parts in `_stream_google()`

**File:** `src/services/assistant/models/model_registry.py:1151-1153`

Gemini 3 sends thinking content as `{"text": "...", "thought": true}`. Currently all `"text"` parts go to content. Check `thought` flag first:

```python
for part in parts:
    if part.get("thought") and "text" in part:
        yield StreamDelta(thinking_content=part["text"])
    elif "text" in part:
        yield StreamDelta(content=part["text"])
    elif "functionCall" in part:
        # existing...
```

### 1.5 `<think>` tag fallback parser

For DashScope configs that embed thinking in content as `<think>...</think>` tags instead of `reasoning_content`, add a stateful parser in `_stream_openai()`. Track `in_think_block` state across chunks since tags may be split across SSE events.

---

## Phase 2: Backend — Agent Loop Thinking Events

### 2.1 Add `thinking_level` to `AgentLoopConfig`

**File:** `src/services/assistant/agent/agent_loop.py:231-296`

```python
thinking_level: str | None = None  # "enabled" for Qwen3, "high"/"medium" for Gemini
```

### 2.2 Pass `thinking_level` in `chat_stream()` call

**File:** `src/services/assistant/agent/agent_loop.py:2030-2035`

```python
async for delta in self.model_registry.chat_stream(
    ...,
    thinking_level=ctx.config.thinking_level,  # NEW
):
```

### 2.3 Emit thinking events in streaming loop

**File:** `src/services/assistant/agent/agent_loop.py:2037-2058`

Add before existing `if delta.content:` block:

- Declare `thinking_started`, `thinking_ended`, `accumulated_thinking` before the `while` loop (~line 1981), reset each iteration
- On `delta.thinking_content`: emit `thinking_start` (once), then `thinking_delta` with text
- When `delta.content` arrives after thinking, or when streaming ends: emit `thinking_end` with full content

The existing event conversion (`assistant_service.py:3175-3202`) already maps `thinking_delta` → `THINKING_DELTA`, `thinking_start` → `THINKING_START`, `thinking_end` → `THINKING_END` — no changes needed there.

### 2.4 Set `thinking_level` when building `AgentLoopConfig`

**File:** `src/services/assistant/assistant_service.py` (where `AgentLoopConfig` is constructed)

```python
thinking_level = None
if "qwen3" in config.model_id.lower():
    thinking_level = "enabled"
elif "gemini-3" in config.model_id:
    thinking_level = "high"
```

---

## Phase 3: Frontend — SSE Event Handling

### 3.1 Add `THINKING_START` to `sse-events.ts`

**File:** `web/src/pages/assistant/sse-events.ts`

```typescript
THINKING_START: "thinking_start",
```

Note: `THINKING_DELTA` and `THINKING_END` already exist.

### 3.2 Extend `ChatMessage` type

**File:** `web/src/pages/assistant/types.ts:269`

Add two fields:

```typescript
streamingThinkingContent?: string;  // in-progress thinking being streamed
isThinkingStreaming?: boolean;       // whether thinking is actively streaming
```

### 3.3 Handle thinking events in `useChatSession`

**File:** `web/src/pages/assistant/hooks/useChatSession.ts` (switch at ~line 896)

Add 3 cases:

- **`THINKING_START`:** set `isThinkingStreaming: true`, clear `streamingThinkingContent`
- **`THINKING_DELTA`:** append `event.data` (string) to `streamingThinkingContent`
- **`THINKING_END`:** set `isThinkingStreaming: false`, move accumulated content to `thinkingContent`, clear `streamingThinkingContent`

Also handle cleanup in `CANCELLED` case.

---

## Phase 4: Frontend — ThinkingPanel Component

### 4.1 Create `ThinkingPanel` component

**New file:** `web/src/pages/assistant/components/ThinkingPanel.tsx`

```typescript
interface ThinkingPanelProps {
  streamingContent?: string;
  finalContent?: string;
  isStreaming: boolean;
}
```

**States:**

1. **Streaming:** Open panel, pulsing brain icon, elapsed timer, scrolling content in `text-muted-foreground` italic style, auto-scroll to bottom
2. **Complete:** Collapsed `<details>`, "Thought for Xs" summary, chevron to expand
3. **Expanded:** Full thinking content in bordered left-margin container

**Styling:** Use existing violet palette: `bg-violet-50/80 dark:bg-violet-900/20`, `text-violet-700 dark:text-violet-300`. Use `Brain` icon from `lucide-react` (already in deps).

### 4.2 Integrate into `ChatMessage.tsx`

**File:** `web/src/pages/assistant/components/ChatMessage.tsx:828-919`

Replace logic:

- `if (isThinkingStreaming || streamingThinkingContent)` → `<ThinkingPanel streaming>`
- `else if (isThinking && no content)` → existing 3-dot animation (non-thinking models fallback)
- `if (thinkingContent && not handled by ThinkingPanel)` → `<ThinkingPanel final>`

Remove the old static `<details>` block (lines 908-919) in favor of `ThinkingPanel`.

---

## Phase 5: Workflow Timeline Enhancement

### 5.1 Add thinking step to `ProcessSummaryBar`

**File:** `web/src/pages/assistant/components/ProcessSummaryBar.tsx`

Add a "Thinking" step as the first entry in the steps timeline when thinking was detected. Shows elapsed time and "Thought for Xs" when complete.

### 5.2 Add thinking timing to `ProcessSummaryState`

**File:** `web/src/pages/assistant/types.ts` (`ProcessSummaryState`)

Add `thinkingDurationMs?: number` field, populated from `THINKING_START`/`THINKING_END` timestamps in `useChatSession`.

---

## Critical Files

| File | Changes |
|------|---------|
| `src/services/assistant/models/model_registry.py` | StreamDelta field, `_stream_openai` extraction, `_build_openai_body` thinking config, `_stream_google` thought parts |
| `src/services/assistant/agent/agent_loop.py` | AgentLoopConfig field, thinking event emission in streaming loop |
| `src/services/assistant/assistant_service.py` | Set `thinking_level` in config construction |
| `web/src/pages/assistant/sse-events.ts` | Add `THINKING_START` |
| `web/src/pages/assistant/types.ts` | `ChatMessage` + `ProcessSummaryState` fields |
| `web/src/pages/assistant/hooks/useChatSession.ts` | 3 new event handlers |
| `web/src/pages/assistant/components/ThinkingPanel.tsx` | New component |
| `web/src/pages/assistant/components/ChatMessage.tsx` | Replace thinking UI |
| `web/src/pages/assistant/components/ProcessSummaryBar.tsx` | Thinking step in timeline |

## Existing Code to Reuse

- `StreamEventType.THINKING_*` enums (`src/models/enums.py:40+`) — already defined
- Event conversion map (`assistant_service.py:3197-3201`) — already maps thinking events
- `SSEEventType.THINKING_DELTA` / `THINKING_END` (`sse-events.ts:11-12`) — already defined
- `ChatMessage.thinkingContent` (`types.ts:269`) — already exists
- `motion` + `AnimatePresence` from `framer-motion` — already in ChatMessage
- `Brain` icon from `lucide-react` — already in AgentPhaseDisplay

## Implementation Order

1. **Phase 1 (1.1–1.3) → Phase 2 (2.1–2.4)** → Backend complete, events flowing
2. **Phase 3 (3.1–3.3)** → Frontend captures events
3. **Phase 4 (4.1–4.2)** → ThinkingPanel visible to user
4. **Phase 1.4** → Gemini thinking (parallel with 3–4)
5. **Phase 5** → Timeline polish

## Verification

1. Start assistant with Qwen 3.6-Plus, send a question requiring reasoning
2. Verify `THINKING_START` / `THINKING_DELTA` / `THINKING_END` events in browser DevTools Network tab (SSE stream)
3. Verify ThinkingPanel appears during TTFT with streaming content
4. Verify panel collapses after thinking ends, shows "Thought for Xs"
5. Verify non-thinking models (qwen-plus, gemini-2.5-flash) still show 3-dot animation
6. Test cancel during thinking — panel should clean up
7. Test multi-iteration tool loop — thinking per iteration
