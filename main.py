from exec_experiment import exec_experiment
from src.utils import read_command_line_args

# Parse arguments
args = read_command_line_args()

exec_experiment(**args.__dict__)