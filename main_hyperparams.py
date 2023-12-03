from src.utils import read_command_line_args
from search_hyperparams import search_hyperparams

# Main
# Parse arguments
args = read_command_line_args()
# Search hyperparameters
search_hyperparams(args)
