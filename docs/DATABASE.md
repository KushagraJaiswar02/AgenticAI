# JARVIS — Database Design

## 1. Storage Strategy

JARVIS uses local persistent storage.

Primary database:

```text
SQLite
```

SQLite stores authoritative structured application data and is accessed through SQLAlchemy.

The initial persistence responsibility is to store conversations, messages, explicit structured memories, tasks, tool executions, settings, migration state, and the metadata needed to reconstruct bounded V0 execution. SQLAlchemy owns database access patterns and parameterized persistence; SQLite remains the authoritative local store.

Semantic memory is accessed through a vector-store abstraction. The initial implementation should prefer a local solution and avoid requiring a remote vector database unless necessary.

## 2. Conceptual Data Model

```text
conversations
    |
    +--- messages

memories
    |
    +--- memory metadata

tasks
    |
    +--- task executions

tool_executions

settings
```

Semantic vectors may reference records in SQLite through stable IDs.

## 3. Suggested Tables

### conversations

- id
- created_at
- updated_at
- title/status if needed

### messages

- id
- conversation_id
- role
- content
- created_at
- metadata

Roles may include:
- user
- assistant
- tool
- system

### memories

- id
- type
- key
- value
- source
- confidence
- created_at
- updated_at
- expires_at nullable
- embedding_reference nullable

Structured memories should be stored as exact data whenever possible.

### tasks

- id
- description
- status
- created_at
- updated_at
- current_step
- metadata

### tool_executions

- id
- task_id nullable
- tool_name
- arguments
- result
- status
- risk_level
- created_at
- duration_ms

### settings

- key
- value
- updated_at

## 4. Memory Rules

Do not store every conversation message as long-term memory.

Short-term conversation remains conversation state.

Long-term memory should be created when:
- the user explicitly asks JARVIS to remember something
- a clearly defined system policy identifies durable information

## 5. Vector Memory

The vector layer should expose operations conceptually equivalent to:

```text
add(id, text, metadata)
search(query, top_k, filters)
delete(id)
```

The vector implementation must not become the source of truth for structured facts.

## 6. Database Requirements

- migrations/versioning
- indexes for common lookups
- foreign keys where appropriate
- timestamps stored consistently
- parameterized SQL
- no secrets in the database
- regular database backup/export capability later

## 7. Initial Vector Decision

Use ChromaDB locally for V0 semantic retrieval. The memory manager must access it through a semantic-memory interface, and embeddings must be provided through a replaceable embedding interface. No application component may treat ChromaDB as authoritative for structured facts.

The vector layer stores semantic representations and stable references to SQLite records. SQLite remains authoritative for structured memory and application state. No remote vector database is required for V0.
