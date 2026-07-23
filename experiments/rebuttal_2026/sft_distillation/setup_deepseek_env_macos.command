#!/bin/zsh
set -euo pipefail

key="$(
  osascript <<'APPLESCRIPT'
try
  display dialog "Paste the DeepSeek API key. The value is hidden and will not be written to a file." default answer "" with hidden answer buttons {"Cancel", "Save"} default button "Save"
  return text returned of result
on error number -128
  return ""
end try
APPLESCRIPT
)"

if [[ -z "$key" ]]; then
  osascript -e 'display alert "No key was saved." as warning'
  exit 1
fi

if [[ ! "$key" =~ '^sk-[A-Za-z0-9_-]{20,}$' ]]; then
  unset key
  osascript -e 'display alert "The value does not look like a DeepSeek API key." as warning'
  exit 2
fi

model="${DEEPSEEK_MODEL:-$(launchctl getenv DEEPSEEK_MODEL)}"
if [[ -z "$model" ]]; then
  model="$(
    osascript <<'APPLESCRIPT'
try
  display dialog "Paste the exact model name returned by the authenticated DeepSeek /models endpoint." default answer "" buttons {"Cancel", "Save"} default button "Save"
  return text returned of result
on error number -128
  return ""
end try
APPLESCRIPT
  )"
fi

if [[ -z "$model" || ! "$model" =~ '^[A-Za-z0-9._/-]+$' ]]; then
  unset key model
  osascript -e 'display alert "No valid model name was saved." as warning'
  exit 3
fi

launchctl setenv DEEPSEEK_API_KEY "$key"
launchctl setenv DEEPSEEK_MODEL "$model"
unset key model

osascript -e 'display dialog "DeepSeek environment configured. Quit and reopen Codex, then return to this task and say: continue." buttons {"OK"} default button "OK"'
