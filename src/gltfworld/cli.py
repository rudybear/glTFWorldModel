"""Command-line entry point for gltfworld.

This is a V0 scaffold: every subcommand is a stub that prints a message and
exits non-zero. Real behavior lands in milestone V1 and later.
"""

import argparse
import sys

_SUBCOMMANDS = ("validate", "inspect", "render", "generate", "stats", "crosscheck")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gltfworld")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name in _SUBCOMMANDS:
        sub = subparsers.add_parser(name)
        sub.add_argument("args", nargs=argparse.REMAINDER)

    parsed = parser.parse_args(argv)

    print(f"gltfworld {parsed.command}: not implemented yet (milestone V1+)")
    return 2


if __name__ == "__main__":
    sys.exit(main())
