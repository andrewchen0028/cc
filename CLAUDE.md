# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python 3.13+ project built with [uv](https://docs.astral.sh/uv/) as the package manager. Currently a minimal starter project with basic structure in place.

## Development Setup

Use `uv` for package management (configured in `pyproject.toml` and `uv.lock`).

### Common Commands

- **Run the main script**: `python main.py`
- **Add dependencies**: `uv add <package-name>`
- **Install dependencies**: `uv sync`

## Project Structure

- `main.py` - Entry point with main() function
- `pyproject.toml` - Project configuration and dependencies
- `uv.lock` - Dependency lock file (auto-managed by uv)

## Current Architecture

The project is in early stages. Currently contains a simple `main()` function that demonstrates basic functionality. As the project grows, update this section with information about major components and module organization.

## Notes

- Python 3.13+ is required
- No external dependencies yet
