# JARVIS — Tool System

## 1. Purpose

Tools are controlled capabilities exposed to the orchestrator and, indirectly, to the LLM.

## 2. Tool Contract

Every tool should define:

```text
name
description
input schema
risk level
confirmation policy
execute()
```

## 3. Initial Tool Groups

### System
- open application
- close application
- system information
- execute approved command

### Files
- list files
- read file
- create file
- move/copy file
- delete file with confirmation

### Browser
- open URL
- search web
- browser navigation

### Memory
- remember
- recall
- remove memory

### Utility
- calculator
- time
- web lookup

## 4. Tool Safety

The LLM cannot bypass tool validation.

Tool arguments must be validated before execution.

High-risk tools must require confirmation.

Never expose arbitrary unrestricted shell execution without a permission boundary.

## 5. Tool Results

Use structured results:

```json
{
  "success": true,
  "data": {},
  "error": null
}
```

or:

```json
{
  "success": false,
  "data": null,
  "error": "reason",
  "retryable": false
}
```
