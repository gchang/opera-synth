import argparse
from pathlib import Path

from synth.config import SimulationInputs


def run():
    """Run the simulation saved in the JSON parameters file."""
    parser = argparse.ArgumentParser(
        description="Run the simulation saved in the JSON parameters file."
    )
    parser.add_argument(
        "-f",
        "--params-file",
        type=Path,
        default=Path("simulation_params.json"),
        help="Paths to simulation parameters JSON file",
    )
    args = parser.parse_args()

    from synth.core import create_simulation_data

    with open(args.params_file) as f:
        inputs = SimulationInputs.model_validate_json(f.read())
        create_simulation_data(inputs)
