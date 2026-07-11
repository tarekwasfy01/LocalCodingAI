<div align="center"><h1>Local Coding AI</h1></div>
<div align="center"><img width="256" height="256" alt="ChatGPT Image 9  Juli 2026, 18_37_58 (1)" src="https://github.com/user-attachments/assets/62aae706-81a3-4a8e-a423-9a575b1ae4bb" /></div>

<div align="center">
<img width="256" height="256" alt="app_icon" src="https://github.com/user-attachments/assets/1e6ff690-d171-4381-a149-36d7c4329727" />
  <h1>Local Coding AI</h1>
  <p>A local Windows coding assistant with multiple Ollama agents, project access, persistent memory, and optional LoRA training.</p>
</div>

> [!WARNING]
> This project is under active development. The application can automatically modify files inside the selected project. Use version control and review important changes.

## Features

- Local chat interface for selected project folders
- GPT-Oss as coordinator and dedicated coding agent
- A team of up to six agents using GPT, Qwen, DeepSeek, StarCoder, OpenClaw, and the primary coder
- Automatic comparison and validation of proposed file changes
- Create, replace, append, and import project files
- Package Python applications as Windows executables
- Formatted and selectable code blocks for PowerShell, Bash, Python, Rust, and other languages
- Persistent local chat, project, and error memory
- Automatically generated JSONL training datasets
- Optional one-step LoRA training after successful responses
- Native Windows GUI without console windows

## Installation

The easiest option is the single-file installer:

```text
DISTRIBUTION/ONE_FILE_INSTALLER/Local Coding AI.exe
```

The installer:

1. installs Local Coding AI under `%LOCALAPPDATA%\Local Coding AI`,
2. creates a desktop shortcut,
3. installs Ollama through `winget` if it is missing,
4. downloads only missing models,
5. configures OpenClaw, and
6. starts the application.

The initial setup downloads several large models. An internet connection and sufficient free disk space are required. Models that are already installed are skipped.

## Models

The following Ollama models are prepared by default:

- `gpt-oss:20b`
- `qwen2.5-coder:7b`
- `qwen2.5-coder:1.5b`
- `deepseek-coder:1.3b`
- `deepseek-coder:6.7b`
- `starcoder2:7b`
- `openclaw-agent:latest` as a locally derived agent model

OpenClaw itself is configured through Ollama's `launch openclaw` integration.

## Usage

1. Start **Local Coding AI**.
2. Create or select a project in the sidebar.
3. Describe the requested change in the chat.
4. The agents inspect the project and generate candidate changes.
5. Only validated actions are applied inside the selected project.

Example request:

```text
Add Open and Save As support for TXT files to the Tkinter text editor.
```

For command-line questions, the application responds directly with formatted code:

```text
Show me the PowerShell command that lists all Python files recursively.
```

## Agent workflow

1. The GPT router analyzes the request, project snapshot, and memory.
2. Coding work is distributed across the configured agent team.
3. Each coding agent returns structured file actions.
4. The manager validates and compares the candidates.
5. Invalid or empty candidates trigger repair and fallback attempts.
6. The safest valid solution is applied.
7. The final result is written to memory and training datasets.

## Supported actions

- `write_file` — create a file or write its complete content
- `replace_text` — replace exactly one matching section
- `append_file` — append content to a file
- `copy_file` — safely import a file into the project
- `package_python_exe` — package a Python script under `dist/`

## Safety rules

- Changes are restricted to the selected project folder.
- Paths containing `..` are blocked.
- Absolute target paths outside the project are rejected.
- `replace_text` runs only when the search text occurs exactly once.
- Ambiguous or unsafe actions are not applied.
- External processes use timeouts.

## Memory and training data

All data is stored locally inside the installation directory:

```text
memory/
  conversations/history.jsonl
  projects/current.md
  agent_traffic/traffic.log

training/
  raw/all_runs.jsonl
  success/runs.jsonl
  errors/runs.jsonl
  fine_tuning/chat_messages.jsonl
  online_adapter/
  short_training_status.log
```

Successful responses are stored in chat fine-tuning format. Failed runs are kept separately so incorrect answers are not automatically treated as valid training examples. Current project memory is loaded by the coordinator during the next request.

### Optional LoRA training

Python and the training dependencies are required for real one-step LoRA updates:

```powershell
pip install -r training_tools\requirements.txt
```

Training runs in the background. Its status and adapter files are stored under `training/`. Training `gpt-oss:20b` requires substantial RAM or VRAM.

## Building from source

Requirements:

- Windows 10 or Windows 11
- A current Rust toolchain with Cargo
- Optional Python installation for EXE packaging and LoRA training

Build the release and distribution:

```powershell
.\BUILD_RELEASE_AND_DISTRIBUTION.bat
```

Check the Rust binaries without building a release:

```powershell
cargo check --bin local_ai_allround_builder
cargo check --bin local_ai_installer
```

Build artifacts are written under `DISTRIBUTION/`. Existing Ollama and model files are preserved during repeated builds instead of being copied again unnecessarily.

## Technology

- Rust
- `eframe` and `egui`
- Ollama HTTP API
- Serde and JSONL
- Python, Transformers, and PEFT for optional LoRA training

## Privacy

Chats, project information, logs, memory, and training data are stored locally. Model requests use the local Ollama API at `127.0.0.1:11434`. Installation and model downloads still contact external package and model sources.

## Project status

Local Coding AI is an experimental local development tool. Review generated changes, back up sensitive projects, and periodically inspect collected training data before production use.

## Screenshots:
<img width="1325" height="1074" alt="Screenshot 2026-07-11 102806" src="https://github.com/user-attachments/assets/08f119d5-6abd-4046-9298-26ea44e5eff7" />

