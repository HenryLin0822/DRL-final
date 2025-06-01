
import numpy as np

from llm import LLMProgramGenerator

from prog_policies.karel import KarelDSL
from prog_policies.minigrid.dsl import MinigridDSL
from prog_policies.karel_tasks import get_task_cls as get_karel_task_cls

KAREL_TASK_MAP_DESC = {
    "StairClimberSparse": "The map is a 12x12 grid surrounded by walls with stairs formed by walls and a marker is randomly initialized on the stairs as a goal.",
    "MazeSparse": "The map is a complex 8x8 grid surrounded by walls and a random marker is placed on an empty cell as a goal.",
    "FourCorners": "The map is an empty 12x12 grid surrounded by walls.",
    "TopOff": "The map is a 12x12 grid surrounded by walls with markers randomly placed on the bottom row of the map.",
    "Harvester": "The map is a 8x8 grid surrounded by walls that starts with a marker on each cell.",
    "CleanHouse": "The map is a complex 14x22 grid made of many connected rooms and is surrounded by walls. There are ten markers randomly placed adjacent to the walls.",
    "DoorKey": "The map is a 8x8 grid surrounded by walls that is vertically split into two chambers. The left chamber is 6x3 grid and the right chamber is 6x2 grid. There is a marker placed randomly on the left chamber as a key, and another marker placed randomly on the right chamber as a goal.",
    "OneStroke": "The map is given by an empty 8x8 grid surrounded by walls.",
    "Seeder": "The map is given by an empty 8x8 grid surrounded by walls.",
    "Snake": "The map is given by an empty 8x8 grid surrounded by walls with a marker randomly placed on the map.",
    "PathFollow": "The map is given by a 8x8 grid surrounded by walls. There is a rugged ascending markers line that starts from the bottom left cell and randomly grows either north or to the east until it reaches the top right cell. Resulting in a rugged markers line connecting the bottom left cell and the top right cell.",
    "WallAvoider": "The map is given by an empty 8x5 grid surrounded by walls."
}

# Task can be chosen from the map above.
TASK         = "StairClimberSparse"
LLM_NUM      = 20
TEMPERATURE  = 2.5
TOP_P        = 0.9

def karel_setup():
    dsl = KarelDSL()
    task_cls = get_karel_task_cls(TASK)
    env_args = {
        "env_height": 8,
        "env_width": 8,
        "crashable": True,
        "leaps_behaviour": True,
        "max_calls": 10000,
    }
    if TASK in ("StairClimber", "StairClimberSparse", "TopOff", "FourCorners"):
        env_args["env_height"] = env_args["env_width"] = 12
    if TASK == "CleanHouse":
        env_args["env_height"], env_args["env_width"] = 14, 22
    if TASK == "WallAvoider":
        env_args["env_height"], env_args["env_width"] = 8, 5

    return task_cls, env_args, dsl


def main():
    print("version v2 0601")
    _, _, dsl = karel_setup()

    generator = LLMProgramGenerator(
        seed=42,
        task=TASK,
        dsl=dsl,
        llm_program_num=LLM_NUM,
        temperature=TEMPERATURE,
        top_p=TOP_P,
    )
    program_nodes, _ = generator.get_program_list_python_to_dsl()
    dsl_programs: list[str] = [dsl.parse_node_to_str(prog) for prog in program_nodes]

    with open(f"dsl_programs_{TASK}.txt", "w") as f:
        for idx, code in enumerate(dsl_programs, 1):
            f.write(f"no_{idx} {code}\n")
    print("files saved.")

if __name__ == "__main__":
    main()