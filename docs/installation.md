# Installation Guide

This guide covers installing Quantuis on Linux, macOS, and Windows.

## Prerequisites

| Requirement | Minimum Version |
|---|---|
| Python | 3.12+ |
| pip | 23.0+ |
| Git | 2.30+ |
| OS | Linux, macOS, or Windows 10+ |

## Linux

### Debian / Ubuntu

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install Python and pip
sudo apt install -y python3 python3-pip python3-venv python3-yaml

# Install Git
sudo apt install -y git
```

### Fedora / RHEL

```bash
sudo dnf install -y python3 python3-pip python3-virtualenv python3-pyyaml git
```

### Arch Linux

```bash
sudo pacman -S python python-pip python-yaml git
```

### Clone and Install

```bash
# Clone the repository
git clone <repo-url>
cd tda

# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Install the package and dependencies
pip install -e .

# Copy the environment template
cp .env.example .env
# Edit .env with your API keys and configuration
nano .env
```

### Systemd Service (Optional)

To run the bot as a background service:

```bash
sudo tee /etc/systemd/system/quantuis.service > /dev/null <<EOF
[Unit]
Description=Quantuis Trading Bot
After=network.target

[Service]
Type=simple
User=$(whoami)
WorkingDirectory=$(pwd)
ExecStart=$(pwd)/venv/bin/python -m crypto_trading_framework.cli live
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable quantuis
sudo systemctl start quantuis
```

### Verify Installation

```bash
python -m crypto_trading_framework.cli version
```

Expected output:

```
Quantuis v5.0.0
```

---

## macOS

### Using Homebrew

```bash
# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python and Git
brew install python git
```

### Clone and Install

```bash
# Clone the repository
git clone <repo-url>
cd tda

# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Install the package and dependencies
pip install -e .

# Copy the environment template
cp .env.example .env
# Edit .env with your API keys and configuration
nano .env
```

### Launchd Service (Optional)

To run the bot as a background service, create a `~/Library/LaunchAgents/com.quantuis.bot.plist` file:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.quantuis.bot</string>
    <key>ProgramArguments</key>
    <array>
        <string>$(pwd)/venv/bin/python</string>
        <string>-m</string>
        <string>crypto_trading_framework.cli</string>
        <string>live</string>
    </array>
    <key>WorkingDirectory</key>
    <string>$(pwd)</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
</dict>
</plist>
```

Then load and start the service:

```bash
launchctl load ~/Library/LaunchAgents/com.quantuis.bot.plist
launchctl start com.quantuis.bot
```

### Verify Installation

```bash
python -m crypto_trading_framework.cli version
```

---

## Windows

### Prerequisites

1. **Python 3.12+** — Download from [python.org](https://www.python.org/downloads/) and ensure "Add Python to PATH" is checked during installation.
2. **Git** — Download from [git-scm.com](https://git-scm.com/download/win) and install with default options.

### Clone and Install

Open PowerShell (Run as Administrator if needed):

```powershell
# Clone the repository
git clone <repo-url>
cd tda

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
.\venv\Scripts\Activate.ps1

# Install the package and dependencies
pip install -e .

# Copy the environment template
copy .env.example .env
# Edit .env with your API keys and configuration
notepad .env
```

### Running as a Windows Service (Optional)

Use NSSM (Non-Sucking Service Manager) to run the bot as a Windows service:

```powershell
# Download NSSM from https://nssm.cc/download
# Extract and add to PATH, then:

nssm install QuantuisBot "python" "-m crypto_trading_framework.cli live"
nssm set QuantuisBot AppDirectory "C:\path\to\tda"
nssm start QuantuisBot
```

### Verify Installation

```powershell
python -m crypto_trading_framework.cli version
```

Expected output:

```
Quantuis v5.0.0
```

---

## Common Setup (All Platforms)

### Environment Variables

Edit `.env` to add your API keys and configuration:

```env
# Exchange API Keys
EXCHANGE_API_KEY=your_api_key_here
EXCHANGE_API_SECRET=your_api_secret_here

# Database (optional, uses in-memory SQLite by default)
DATABASE_URL=postgresql://user:password@localhost:5432/quantuis

# Redis (optional)
REDIS_URL=redis://localhost:6379/0

# Telegram Alerts (optional)
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

### Configuration Files

Quantuis uses a layered config system:

| File | Purpose |
|---|---|
| `config/base.yaml` | Shared defaults for all environments |
| `config/overlays/crypto.yaml` | Crypto-specific overrides |
| `config/overlays/idx.yaml` | IDX (Indonesian stock index) overrides |
| `.env` | Secrets and sensitive values |

### Running the Bot

```bash
# Dry-run backtest
python -m crypto_trading_framework.cli backtest

# Live trading (dry_run mode by default)
python -m crypto_trading_framework.cli live

# Launch observability cockpit
python -m crypto_trading_framework.cli cockpit

# Generate signals
python -m crypto_trading_framework.cli signals

# Check version
python -m crypto_trading_framework.cli version
```

### Running the FastAPI Server

```bash
# Start the API server
uvicorn main:app --host 0.0.0.0 --port 8000
```

Or via the CLI:

```bash
python -m crypto_trading_framework.cli cockpit
```

### Docker (All Platforms)

```bash
# Start full stack (app + Redis + TimescaleDB + Grafana)
make docker-up

# Stop
make docker-down
```

## Troubleshooting

### `python` command not found

- **Linux**: `sudo apt install python3`
- **macOS**: `brew install python`
- **Windows**: Reinstall Python and check "Add Python to PATH"

### `pip` command not found

- **Linux**: `sudo apt install python3-pip`
- **macOS**: `pip3 install --upgrade pip`
- **Windows**: `python -m pip install --upgrade pip`

### Permission denied on venv activation

- **Linux/macOS**: `chmod +x venv/bin/activate`
- **Windows**: Run PowerShell as Administrator, or set execution policy:
  ```powershell
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
  ```

### Module not found after `pip install -e .`

Ensure the virtual environment is activated:

```bash
source venv/bin/activate   # Linux/macOS
.\venv\Scripts\Activate.ps1  # Windows
```

### uvicorn import error

If `uvicorn main:app` fails with an import error, ensure `main.py`
exists at the project root and contains a valid FastAPI `app` object.
The current `main.py` includes both the CLI wrapper and the FastAPI app.