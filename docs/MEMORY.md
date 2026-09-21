# JARVIS — Memory System

## 1. Memory Types

### Working Memory
Current request and active task state.

### Conversation Memory
Messages within a conversation/session.

### Long-Term Structured Memory
Stable facts explicitly worth retaining.

Example:

```text
primary_project = EventX
```

### Semantic Memory
Unstructured information retrievable through semantic similarity.

## 2. What Should Be Remembered

Good candidates:
- explicit user instructions to remember
- stable project facts
- durable preferences
- important recurring configuration

Do not automatically convert every statement into long-term memory.

## 3. Retrieval

Relevant memories should be retrieved before LLM reasoning when they can materially affect the request.

Avoid dumping the entire memory database into the prompt.

## 4. Memory Lifecycle

Future versions should support:
- creation
- update
- correction
- deletion
- expiration where appropriate
- provenance

## 5. Privacy

Memory is local by default where practical.

Secrets, credentials, tokens, and highly sensitive information should not be casually persisted as normal memory.
