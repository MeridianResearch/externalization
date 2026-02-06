---
name: run-code
description: Execute Python code autonomously by writing to .ai-code/code/ directory
allowed-tools: Bash(python *)
---

# Run Code Skill

Execute Python code by writing scripts to `.ai-code/code/` and running them.

## Instructions

1. **Write the code** to `.ai-code/code/<script_name>.py`
2. **Execute** with: `python .ai-code/code/<script_name>.py`
3. **Report** the output or fix any errors

## Usage

When the user asks to run or test code:
- Write the Python script to `.ai-code/code/`
- Run it immediately without asking for permission
- Show the output and interpret results

## Arguments

Any text after `/run-code` is the task description for what code to write and execute.

## Example

```
/run-code load the sft_think_v1 model and print its config
```

This will write a script to load the model and display its configuration.
