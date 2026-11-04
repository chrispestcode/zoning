import sys
from loguru import logger
from datetime import date



if __name__ == '__main__':
    if len(sys.argv) > 1:
        command = sys.argv[1]
        results = search_and_print_results(command)
        print(f"Querying: {command}")
    else:
        print("No command provided")
