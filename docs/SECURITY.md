# JARVIS — Security Model

## 1. Threat Model

JARVIS can eventually control the user's computer. Therefore, model mistakes, malicious content, prompt injection, and accidental destructive actions are security concerns.

## 2. Core Rules

1. LLM output is untrusted.
2. Tool inputs are validated.
3. Permissions are enforced outside the LLM.
4. Destructive actions require confirmation.
5. Secrets never enter prompts unnecessarily.
6. Secrets are never logged.
7. External web content must not automatically gain tool authority.
8. Windows/system commands are limited to an explicit allowlist executed through Python `subprocess`.
9. Filesystem tools operate only inside explicitly configured safe filesystem boundaries.
10. Destructive confirmation is bound to the exact operation and target.
11. Secrets must never enter prompts, logs, or long-term memory.

## 3. Prompt Injection

Web pages, documents, emails, and other external content may contain instructions designed to manipulate the model.

External content is data, not authority.

A retrieved instruction must not override JARVIS system policies or tool permissions.

The LLM is an untrusted proposer. It cannot grant itself capabilities, bypass validation, approve operations, or execute tools directly. External web content returned through Playwright or web lookup is data, never authority.

## 4. Shell Access

Arbitrary shell execution is prohibited.

Initial implementation uses Python `subprocess` only for explicitly allowlisted commands with validated arguments. pywinauto may be used for supported GUI/application automation where required. No unrestricted shell execution is exposed.

## 5. Filesystem Access

The assistant operates only inside explicitly defined safe filesystem paths by default. The safe paths and path-validation policy must be configured before filesystem tools are enabled.

Unrestricted filesystem access is prohibited. Destructive operations require target-specific confirmation.

## 6. Confirmation

Confirmation should be tied to the actual operation and target.

Example:

```text
This will permanently delete:
C:\PROJECTSPACE\eventx\test.txt

Continue? [y/N]
```

Confirmation must be requested after validation and must identify the actual operation and target. Approval for one target cannot authorize a different target.

## 7. Future Security Work

- capability-based permissions
- sandboxing
- audit logs
- configurable trusted paths
- per-tool authorization
- secure credential storage

For V0, secrets are supplied through environment configuration and must not be copied into prompts, logs, or memory.
