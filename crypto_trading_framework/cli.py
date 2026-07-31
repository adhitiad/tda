from __future__ import annotations

import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel

app = typer.Typer(
    name="tda",
    help="Quantuis - Trading Data Analysis Framework",
    add_completion=False,
)
console = Console()


def _load_config(
    config_path: str,
    overlay_path: str | None,
) -> dict:
    from crypto_trading_framework.config.settings import load_config

    return load_config(config_path=config_path, overlay_path=overlay_path)


@app.command()
def backtest(
    config: str = typer.Option("config/base.yaml", "--config", "-c", help="Path to base config YAML"),
    overlay: str | None = typer.Option(None, "--overlay", "-o", help="Path to overlay config YAML"),
    symbol: str | None = typer.Option(None, "--symbol", "-s", help="Specific symbol to backtest"),
) -> None:
    """Run a backtest."""
    from crypto_trading_framework.ml.backtest import BacktestEngine

    cfg = _load_config(config, overlay)
    engine = BacktestEngine(cfg)
    if symbol:
        engine.run_symbol(symbol)
    else:
        engine.run_all()


@app.command()
def live(
    config: str = typer.Option("config/base.yaml", "--config", "-c", help="Path to base config YAML"),
    overlay: str | None = typer.Option(None, "--overlay", "-o", help="Path to overlay config YAML"),
) -> None:
    """Start the live trading bot."""
    from crypto_trading_framework.core.bot import AutomatedTradingBot

    cfg = _load_config(config, overlay)
    bot = AutomatedTradingBot(cfg)
    bot.start()


@app.command()
def cockpit(
    config: str = typer.Option("config/base.yaml", "--config", "-c", help="Path to base config YAML"),
    overlay: str | None = typer.Option(None, "--overlay", "-o", help="Path to overlay config YAML"),
) -> None:
    """Launch the Streamlit observability cockpit."""
    import subprocess

    cfg = _load_config(config, overlay)
    cockpit_script = Path(__file__).parent / "observability" / "cockpit.py"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(cockpit_script),
            "--",
            "--config",
            cfg.get("paths", {}).get("config_file", "config/base.yaml"),
        ],
        check=False,
    )


@app.command()
def signals(
    config: str = typer.Option("config/base.yaml", "--config", "-c", help="Path to base config YAML"),
    overlay: str | None = typer.Option(None, "--overlay", "-o", help="Path to overlay config YAML"),
) -> None:
    """Generate and display trading signals."""
    cfg = _load_config(config, overlay)
    console.print(Panel("[bold cyan]Quantuis Signal Generator[/bold cyan]"))
    console.print(f"Config loaded from {cfg.get('paths', {}).get('config_file', 'config/base.yaml')}")
    console.print(f"Symbols: {', '.join(cfg.get('data', {}).get('symbols', []))}")


@app.command()
def train(
    config: str = typer.Option("config/base.yaml", "--config", "-c", help="Path to base config YAML"),
    overlay: str | None = typer.Option(None, "--overlay", "-o", help="Path to overlay config YAML"),
) -> None:
    """Train models for all symbols."""
    console.print(Panel("[bold cyan]Quantuis Model Training[/bold cyan]"))
    console.print("Training not yet implemented as standalone command.")


@app.command()
def version() -> None:
    """Print version information."""
    from crypto_trading_framework import __version__

    console.print(f"[bold]Quantuis[/bold] v{__version__}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
