import click


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """A simple weather CLI tool."""
    pass


@cli.command()
@click.argument("location")
def forecast(location: str):
    """Get weather forecast for a location."""
    click.echo(f"Fetching forecast for {location}...")


@cli.command()
@click.argument("location")
def current(location: str):
    """Get current weather for a location."""
    click.echo(f"Fetching current weather for {location}...")


if __name__ == "__main__":
    cli()
