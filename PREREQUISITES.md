# forge — Prerequisites & Local LLM Setup Guide

Everything you need to install and configure **before** running the forge install command.
This guide covers all hardware tiers so you can pick the right model for your machine.

---

## What forge needs to run

| Component | What it is | Cost |
|---|---|---|
| **Python 3.10+** | Runs forge itself | Free |
| **LM Studio** | Desktop app that runs your local AI model | Free |
| **A local LLM** | The coding model (downloaded inside LM Studio) | Free |
| **Gemini API key** | The "master brain" that plans and reviews | Free tier |

---

## Step 1 — Check your hardware

forge's local worker runs entirely on your machine. The right model depends on your GPU.
Open your system info and find your GPU and its VRAM amount.

**Windows:** Task Manager → Performance → GPU  
**macOS:** Apple menu → About This Mac → Graphics  
**Linux:** `nvidia-smi` or `lspci | grep VGA`

Then find your tier below:

---

### Tier 1 — 6–8 GB VRAM (RTX 3060 / RTX 4060 / RX 6600 and similar)

This is the most common laptop GPU range. You get full GPU acceleration — fast and responsive.

**Recommended model:** `Qwen3.5-9B-Instruct-Q4_K_M`  
**VRAM used:** ~5.5 GB  
**Speed:** ~40–55 tokens/second  
**Quality:** Excellent for everyday coding tasks  

**Alternative if Qwen3.5-9B isn't available yet in LM Studio:**  
`Qwen2.5-Coder-7B-Instruct-Q4_K_M` — same speed, slightly older, still very capable

**What to avoid at this tier:**  
Any model above 14B parameters — it will spill to RAM and drop to ~8 tokens/second,
which is too slow for an agent loop.

---

### Tier 2 — 10–12 GB VRAM (RTX 3080 / RTX 4070 / M1 Pro 16 GB and similar)

You have room for a larger model with meaningfully better reasoning.

**Recommended model:** `Qwen2.5-Coder-14B-Instruct-Q4_K_M`  
**VRAM used:** ~9 GB  
**Speed:** ~30–40 tokens/second  
**Quality:** Strong — handles complex multi-file refactors well  

**Alternative:** `DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M` (16B MoE, only 2.4B active — fits well)

---

### Tier 3 — 16–24 GB VRAM (RTX 3090 / RTX 4080 / RTX 4090 / M2 Max and similar)

Full GPU fit for large models. This is where local coding agents get genuinely impressive.

**Recommended model:** `Qwen2.5-Coder-32B-Instruct-Q4_K_M`  
**VRAM used:** ~20 GB  
**Speed:** ~25–35 tokens/second  
**Quality:** Near frontier — handles architecture-level decisions independently  

**Alternative:** `DeepSeek-Coder-V2-Instruct-Q4_K_M` (236B MoE, 21B active — excellent)

---

### Tier 4 — No dedicated GPU / CPU only / 4 GB VRAM

You can still run forge, just with a smaller model and slower speeds.

**Recommended model:** `Qwen2.5-Coder-3B-Instruct-Q4_K_M`  
**RAM used:** ~2.5 GB  
**Speed:** ~8–15 tokens/second (CPU), ~20 tokens/second (4 GB GPU)  
**Quality:** Good for small, focused tasks. Struggles with large refactors.  

**Tip:** On CPU, set `forge auto --checkpoint-every 3` so you review more frequently
and catch issues before they compound across many slow generations.

---

### Apple Silicon (M1 / M2 / M3 / M4 — all variants)

Apple Silicon uses unified memory — your RAM is your VRAM. This is a big advantage.

| Chip | RAM | Recommended model |
|---|---|---|
| M1 / M2 (8 GB) | 8 GB | `Qwen3.5-9B-Instruct-Q4_K_M` |
| M1 Pro / M2 Pro (16 GB) | 16 GB | `Qwen2.5-Coder-14B-Instruct-Q4_K_M` |
| M1 Max / M2 Max (32 GB) | 32 GB | `Qwen2.5-Coder-32B-Instruct-Q4_K_M` |
| M2 Ultra / M3 Max (64 GB+) | 64 GB+ | `Qwen2.5-Coder-32B-Instruct-Q5_K_M` |

LM Studio has native Apple Silicon support — models run via Metal, not CPU.
Performance is excellent across all M-series chips.

---

## Step 2 — Install Python 3.10 or higher

### Windows

1. Go to **https://python.org/downloads**
2. Download the latest Python 3.12 installer (`.exe`)
3. Run it — **check "Add Python to PATH"** before clicking Install
4. Verify: open Command Prompt and run:
   ```
   python --version
   ```
   Should show `Python 3.12.x` or similar

### macOS

Option A — Homebrew (recommended if you have it):
```bash
brew install python@3.12
python3 --version
```

Option B — Official installer:
1. Go to **https://python.org/downloads/macos**
2. Download and run the `.pkg` installer
3. Verify: open Terminal and run `python3 --version`

### Linux (Ubuntu / Debian)
```bash
sudo apt update
sudo apt install python3.12 python3.12-venv python3-pip -y
python3 --version
```

### Linux (Fedora / RHEL)
```bash
sudo dnf install python3.12 -y
python3 --version
```

**Minimum required:** Python 3.10. Python 3.11 or 3.12 recommended.

---

## Step 3 — Install LM Studio

LM Studio is the desktop app that downloads and serves your local model.
It handles all the GPU driver complexity for you.

### Download

Go to **https://lmstudio.ai** and download for your OS:
- **Windows:** `.exe` installer
- **macOS:** `.dmg` (Universal — works on Intel and Apple Silicon)
- **Linux:** `.AppImage`

### Linux AppImage setup
```bash
chmod +x LM_Studio-*.AppImage
./LM_Studio-*.AppImage
```

### First launch

When LM Studio opens you will see a search bar at the top. This is where you
search for and download models. You do not need an account.

---

## Step 4 — Download your local model in LM Studio

1. Open LM Studio
2. Click the **Search** tab (magnifying glass icon on the left)
3. Search for the model name from your tier above, for example: `Qwen3.5-9B`
4. Find the result from **Qwen** (the publisher) — look for the Q4_K_M quantization
5. Click **Download** — the file size will be shown (typically 4–20 GB depending on model)
6. Wait for the download to complete (progress bar at the bottom)

**Which quantization to pick if you see multiple options:**

| Quantization | VRAM impact | Quality | Pick when |
|---|---|---|---|
| Q8_0 | Highest | Best | You have plenty of VRAM headroom |
| Q5_K_M | High | Very good | You have 2–3 GB spare VRAM |
| Q4_K_M | Medium | Good | **Default choice — best balance** |
| Q3_K_M | Low | Acceptable | Tight on VRAM |
| Q2_K | Lowest | Degraded | Last resort only |

For forge, **Q4_K_M is the recommended default** at any tier.

---

## Step 5 — Start the LM Studio local server

After the model downloads:

1. Click the **Local Server** tab (the `<->` icon on the left sidebar)
2. In the model dropdown at the top, select your downloaded model
3. Set these options before starting:

   | Setting | Value | Why |
   |---|---|---|
   | Context Length | `8192` | Matches forge's 8K window — do not increase on 6–8 GB VRAM |
   | GPU Offload | Max (slider all the way right) | Full GPU acceleration |
   | Temperature | `0.6` | Balanced creativity/precision for coding |

4. Click **Start Server** (green button)
5. You should see: `Server running on http://localhost:1234`

**Verify it works** — open your browser and go to:
```
http://localhost:1234/v1/models
```
You should see a JSON response listing your model. If you do, LM Studio is ready.

> **Important:** LM Studio must be running with the server started every time you
> use forge. You can leave it running in the background — it uses no GPU when idle.

---

## Step 6 — Get a free Gemini API key

Gemini is forge's master brain — it plans your project and reviews code quality.
The free tier is generous enough for personal use.

1. Go to **https://aistudio.google.com**
2. Sign in with your Google account
3. Click **Get API key** in the top left
4. Click **Create API key**
5. Copy the key — it starts with `AIza` and is about 39 characters long

**Free tier limits (as of 2025):**
- Gemini 2.0 Flash: 1,500 requests/day, 15 requests/minute
- Gemini 2.5 Flash: 500 requests/day, 10 requests/minute

For a personal coding agent running `forge auto`, 1,500 requests/day is more than enough
for several full project builds per day.

**Keep this key ready** — the `forge setup` wizard will ask you to paste it.
You do not need to set any environment variables manually.

---

## Step 7 — Install pipx (recommended) or verify pip

pipx installs Python CLI tools in isolated environments — it prevents dependency
conflicts between forge and other Python tools on your machine.

### Windows
```
pip install pipx
pipx ensurepath
```
Then restart your terminal.

### macOS
```bash
brew install pipx
pipx ensurepath
```

### Linux
```bash
pip install pipx --user
pipx ensurepath
```
Then restart your terminal or run `source ~/.bashrc`.

**Verify:**
```bash
pipx --version
```

If you prefer not to use pipx, regular `pip install forge-agent` also works —
pipx is just cleaner for CLI tools.

---

## Pre-flight checklist

Before running the forge install command, confirm each item:

```
[ ] Python 3.10 or higher installed
      python --version  →  Python 3.10.x or higher

[ ] LM Studio installed and open

[ ] Local model downloaded in LM Studio (Q4_K_M recommended for your tier)

[ ] LM Studio local server started on port 1234
      http://localhost:1234/v1/models  →  returns JSON in browser

[ ] Gemini API key copied (starts with AIza...)
      Have it ready to paste — forge setup will ask for it

[ ] pipx installed (optional but recommended)
      pipx --version  →  any version number
```

All five checked? You are ready to install forge.

---

## Install forge

```bash
# Recommended
curl -fsSL https://raw.githubusercontent.com/Tushaarxr/forge/main/install.sh | bash

# Or with pipx directly
pipx install git+https://github.com/Tushaarxr/forge.git

# Or with pip
pip install git+https://github.com/Tushaarxr/forge.git

# Or with Docker (no Python install needed)
docker run -it \
  -e GEMINI_API_KEY=AIza... \
  -v $(pwd):/workspace \
  tushaarxr/forge-agent auto "build a FastAPI todo app"
```

After install, run the setup wizard:
```bash
forge setup
```

The wizard will verify your LM Studio connection, ask for your Gemini key,
confirm your model choice, and initialize your first project — all in one flow.

---

## Troubleshooting

**`forge: command not found` after install**  
Run `pipx ensurepath` and restart your terminal.
On Windows, ensure Python's Scripts directory is in your PATH.

**LM Studio server not detected during `forge setup`**  
Confirm the server is started (green button in Local Server tab, not just the app open).
Check `http://localhost:1234/v1/models` in your browser — it must return JSON.

**Model generates very slowly (under 10 tokens/second)**  
GPU offload is not fully active. In LM Studio Local Server tab, slide GPU Offload
all the way to the right, stop the server, and restart it.
Check that your GPU drivers are up to date.

**Gemini API key rejected**  
Keys sometimes take 1–2 minutes to activate after creation in AI Studio.
Wait a moment and try `forge setup --key` to re-enter just the key.

**`pip install forge-agent` fails with permission error on Linux/macOS**  
Use `pip install forge-agent --user` or switch to pipx.

**Context length warning during `forge auto`**  
Set Context Length to exactly `8192` in LM Studio (not higher).
On 6–8 GB VRAM cards, higher context lengths cause RAM spill which tanks speed.

---

## What model name to put in forge setup

When `forge setup` asks for your local model name, use the exact string shown
in LM Studio's Local Server tab model dropdown. It will look like one of these:

```
qwen3.5-9b-instruct
Qwen3.5-9B-Instruct-Q4_K_M
qwen2.5-coder-7b-instruct
Qwen2.5-Coder-14B-Instruct-Q4_K_M
```

Copy it exactly — capitalisation and hyphens matter.
forge setup will auto-detect available models from LM Studio and let you pick
from a list, so you typically will not need to type this manually.
