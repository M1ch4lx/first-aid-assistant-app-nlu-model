from init_bot_files import init_project
from nlu_only_run import run_nlu
from run_bot import run_bot
from nlu_train import train
import argparse
from export_to_mobile import export_project
from server import start_server

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Conversational Bot Builder CLI")
    parser.add_argument("command", choices=["train", "nlu", "run", "init", "export", "start_server"], help="Komenda do wykonania")
    
    args = parser.parse_args()
    
    if args.command == "init":
        init_project()
    elif args.command == "train":
        train()
    elif args.command == "nlu":
        run_nlu()
    elif args.command == "run":
        run_bot()
    elif args.command == "export":
        export_project()
    elif args.command == "start_server":
        start_server()