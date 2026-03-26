# Universal Tool for Implementing a Conversational System Backend

## Requirements

- Python 3.11
- Conda / UV (Optional)

## Installing Libraries

For Python Venv or Conda: `run pip install -r requirements.txt`
For UV: run `uv pip install -r requirements.txt`

## First Aid Assistant Application

This tool is not limited to just creating a web service for a first aid assistant. It can be used to build conversational systems for any purpose.

In the root directory of the project, there is a `bot` folder containing the assistant files for the first aid assistant built with this tool.

To start the web service that the application will use, simply run the command `python bot.py start_server` in the root directory.

## Building Your Own Conversational System

1. Remove the `bot` folder from the root directory (if it exists).
2. Create a new project by running `python bot.py init`.
3. Files with the `.yml` extension define the behavior of the system.
4. After modifying the `nlu.yml` file, train the NLU model by running `python bot.py train`.
5. To test only the intent classification (the NLU model itself), use `python bot.py nlu`.
6. To test the system locally, run `python bot.py run`.
7. If the system works as expected, you can run it on a server by executing `python bot.py start_server`.
