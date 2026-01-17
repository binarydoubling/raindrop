"""Shell completion generation command."""

import click
from rich.console import Console

console = Console()

BASH_COMPLETION = """# Bash completion for raindrop
# Add to ~/.bashrc or ~/.bash_completion

_raindrop_completion() {
    local IFS=$'\\n'
    local response

    response=$(env COMP_WORDS="${COMP_WORDS[*]}" COMP_CWORD=$COMP_CWORD _RAINDROP_COMPLETE=bash_complete $1)

    for completion in $response; do
        IFS=',' read type value <<< "$completion"
        COMPREPLY+=($value)
    done

    return 0
}

_raindrop_completion_setup() {
    complete -o nosort -F _raindrop_completion raindrop
}

_raindrop_completion_setup
"""

ZSH_COMPLETION = """#compdef raindrop
# Zsh completion for raindrop
# Add to ~/.zshrc or place in fpath

_raindrop() {
    local -a commands
    commands=(
        'current:Get current weather for a location'
        'hourly:Show hourly forecast with deltas'
        'daily:Show daily forecast with technical analysis'
        'aqi:Show air quality index and pollutants'
        'alerts:Show active weather alerts and warnings'
        'astro:Show astronomical data (sun, moon, golden hour)'
        'clothing:Get clothing suggestions based on weather'
        'compare:Compare weather across multiple locations'
        'config:View and manage settings'
        'discussion:Show NWS Area Forecast Discussion'
        'fav:Manage favorite locations'
        'history:Compare today with past years'
        'precip:Show precipitation forecast'
        'route:Show weather conditions along a route'
    )
    
    local -a config_subcommands
    config_subcommands=(
        'show:Show current settings'
        'set:Set a configuration value'
        'unset:Unset a configuration value'
        'models:List available weather models'
    )
    
    local -a fav_subcommands
    fav_subcommands=(
        'list:List all saved favorites'
        'add:Add a favorite location'
        'remove:Remove a favorite location'
    )
    
    _arguments -C \\
        '--version[Show version]' \\
        '--help[Show help]' \\
        '1: :->command' \\
        '*:: :->args'
    
    case $state in
        command)
            _describe -t commands 'raindrop commands' commands
            ;;
        args)
            case $words[1] in
                config)
                    _describe -t subcommands 'config subcommands' config_subcommands
                    ;;
                fav)
                    _describe -t subcommands 'fav subcommands' fav_subcommands
                    ;;
                current|hourly|daily|aqi|alerts|astro|clothing|discussion|history|precip)
                    _arguments \\
                        '1:location:' \\
                        '-c[Country code]:country:' \\
                        '--country[Country code]:country:' \\
                        '--json[Output as JSON]' \\
                        '--help[Show help]'
                    ;;
                compare)
                    _arguments \\
                        '*:locations:' \\
                        '--json[Output as JSON]' \\
                        '--help[Show help]'
                    ;;
                route)
                    _arguments \\
                        '1:origin:' \\
                        '2:destination:' \\
                        '-s[Number of stops]:stops:' \\
                        '--stops[Number of stops]:stops:' \\
                        '-d[Departure time]:time:' \\
                        '--depart[Departure time]:time:' \\
                        '--speed[Average speed]:speed:' \\
                        '--json[Output as JSON]' \\
                        '--help[Show help]'
                    ;;
            esac
            ;;
    esac
}

_raindrop "$@"
"""

FISH_COMPLETION = """# Fish completion for raindrop
# Save to ~/.config/fish/completions/raindrop.fish

# Disable file completion by default
complete -c raindrop -f

# Main commands
complete -c raindrop -n __fish_use_subcommand -a current -d 'Get current weather'
complete -c raindrop -n __fish_use_subcommand -a hourly -d 'Show hourly forecast'
complete -c raindrop -n __fish_use_subcommand -a daily -d 'Show daily forecast'
complete -c raindrop -n __fish_use_subcommand -a aqi -d 'Show air quality'
complete -c raindrop -n __fish_use_subcommand -a alerts -d 'Show weather alerts'
complete -c raindrop -n __fish_use_subcommand -a astro -d 'Show astronomical data'
complete -c raindrop -n __fish_use_subcommand -a clothing -d 'Get clothing suggestions'
complete -c raindrop -n __fish_use_subcommand -a compare -d 'Compare locations'
complete -c raindrop -n __fish_use_subcommand -a config -d 'Manage settings'
complete -c raindrop -n __fish_use_subcommand -a discussion -d 'Show NWS discussion'
complete -c raindrop -n __fish_use_subcommand -a fav -d 'Manage favorites'
complete -c raindrop -n __fish_use_subcommand -a history -d 'Historical comparison'
complete -c raindrop -n __fish_use_subcommand -a precip -d 'Precipitation forecast'
complete -c raindrop -n __fish_use_subcommand -a route -d 'Route weather'

# Config subcommands
complete -c raindrop -n "__fish_seen_subcommand_from config" -a show -d 'Show settings'
complete -c raindrop -n "__fish_seen_subcommand_from config" -a set -d 'Set a value'
complete -c raindrop -n "__fish_seen_subcommand_from config" -a unset -d 'Unset a value'
complete -c raindrop -n "__fish_seen_subcommand_from config" -a models -d 'List models'

# Fav subcommands
complete -c raindrop -n "__fish_seen_subcommand_from fav" -a list -d 'List favorites'
complete -c raindrop -n "__fish_seen_subcommand_from fav" -a add -d 'Add favorite'
complete -c raindrop -n "__fish_seen_subcommand_from fav" -a remove -d 'Remove favorite'

# Common options
complete -c raindrop -s c -l country -d 'Country code (e.g., US)'
complete -c raindrop -l json -d 'Output as JSON'
complete -c raindrop -l help -d 'Show help'
complete -c raindrop -l version -d 'Show version'

# Hourly options
complete -c raindrop -n "__fish_seen_subcommand_from hourly" -s n -l hours -d 'Number of hours'
complete -c raindrop -n "__fish_seen_subcommand_from hourly" -l spark -d 'Show sparkline'

# Daily options
complete -c raindrop -n "__fish_seen_subcommand_from daily" -s n -l days -d 'Number of days'

# Route options
complete -c raindrop -n "__fish_seen_subcommand_from route" -s s -l stops -d 'Number of stops'
complete -c raindrop -n "__fish_seen_subcommand_from route" -s d -l depart -d 'Departure time'
complete -c raindrop -n "__fish_seen_subcommand_from route" -l speed -d 'Average speed (mph)'
"""


@click.group()
def completions():
    """Generate shell completion scripts."""
    pass


@completions.command("bash")
def completion_bash():
    """Generate Bash completion script.

    \b
    Usage:
      raindrop completions bash >> ~/.bashrc
      # or
      raindrop completions bash > /etc/bash_completion.d/raindrop
    """
    click.echo(BASH_COMPLETION)


@completions.command("zsh")
def completion_zsh():
    """Generate Zsh completion script.

    \b
    Usage:
      raindrop completions zsh > ~/.zfunc/_raindrop
      # Add to ~/.zshrc: fpath+=~/.zfunc
    """
    click.echo(ZSH_COMPLETION)


@completions.command("fish")
def completion_fish():
    """Generate Fish completion script.

    \b
    Usage:
      raindrop completions fish > ~/.config/fish/completions/raindrop.fish
    """
    click.echo(FISH_COMPLETION)


@completions.command("install")
@click.argument("shell", type=click.Choice(["bash", "zsh", "fish"]))
def completion_install(shell: str):
    """Show installation instructions for completions."""
    if shell == "bash":
        console.print("\n[bold]Bash Completion Installation[/bold]\n")
        console.print("Option 1: Add to your bashrc:")
        console.print("  [cyan]raindrop completions bash >> ~/.bashrc[/cyan]")
        console.print("  [cyan]source ~/.bashrc[/cyan]\n")
        console.print("Option 2: System-wide (requires sudo):")
        console.print(
            "  [cyan]raindrop completions bash | sudo tee /etc/bash_completion.d/raindrop[/cyan]\n"
        )

    elif shell == "zsh":
        console.print("\n[bold]Zsh Completion Installation[/bold]\n")
        console.print("1. Create completion directory:")
        console.print("  [cyan]mkdir -p ~/.zfunc[/cyan]\n")
        console.print("2. Generate completion file:")
        console.print("  [cyan]raindrop completions zsh > ~/.zfunc/_raindrop[/cyan]\n")
        console.print("3. Add to ~/.zshrc (before compinit):")
        console.print("  [cyan]fpath+=~/.zfunc[/cyan]\n")
        console.print("4. Reload shell:")
        console.print("  [cyan]exec zsh[/cyan]\n")

    elif shell == "fish":
        console.print("\n[bold]Fish Completion Installation[/bold]\n")
        console.print("Run:")
        console.print(
            "  [cyan]raindrop completions fish > ~/.config/fish/completions/raindrop.fish[/cyan]\n"
        )
        console.print("Completions will be available in new fish sessions.\n")
