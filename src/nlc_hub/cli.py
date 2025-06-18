#!/usr/bin/env python3
import typer

app = typer.Typer(help="NLC-Hub 命令行工具")

@app.command()
def train(config: str = typer.Option(..., help="训练配置文件路径")):
    """启动训练流程"""
    from nlc_hub.train import tunner
    tunner.main(config)

@app.command()
def webui(host: str = typer.Option("0.0.0.0", help="WebUI 监听地址"), port: int = typer.Option(7860, help="WebUI 端口")):
    """启动 Web UI 服务"""
    typer.echo(f"启动 WebUI，监听地址: {host}:{port}")

if __name__ == "__main__":
    app()
