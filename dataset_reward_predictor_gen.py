import os
import json
import re
import torch
from environments.karel_env import KarelEnvironment
from models.program_executor import ProgramExecutor

token_map = {
    0: 'DEF', 1: 'run', 2: 'm(', 3: 'm)',
    4: 'move', 5: 'turnLeft', 6: 'turnRight',
    7: 'pickMarker', 8: 'putMarker',
    9: 'REPEAT', 10: 'r(', 11: 'r)',
    12: 'R=2', 13: 'R=3', 14: 'R=4', 15: 'R=5',
    16: 'IF', 17: 'IFELSE', 18: 'ELSE',
    19: 'i(', 20: 'i)', 21: 'e(', 22: 'e)',
    23: 'frontIsClear', 24: 'leftIsClear', 25: 'rightIsClear',
    26: 'markersPresent', 27: 'noMarkersPresent',
    28: 'not', 29: 'c(', 30: 'c)',
    31: 'WHILE', 32: 'w(', 33: 'w)',
}
reverse_token_map = {v: k for k, v in token_map.items()}


def extract_program_strings(filename):
    programs = []
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        with open(filename, 'r', encoding='latin-1') as f:
            lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue
        idx = line.find('DEF')
        if idx == -1:
            continue
        program_str = line[idx:]
        programs.append(program_str)
    return programs


def tokenize_program(program_str):
    tokens = program_str.split()
    token_ids = []
    for tok in tokens:
        if tok not in reverse_token_map:
            raise ValueError(f"Unknown token: '{tok}'.  Please check token map\nprogram: {program_str}")
        token_ids.append(reverse_token_map[tok])
    return token_ids


def main():
    folder_path = './original_dataset/data'         
    for filename in os.listdir(folder_path):
        if not filename.endswith('.txt'):
            continue
        pattern = re.compile(r'^id_(\d+)\.txt$')
        id = pattern.match(filename).group(1)
        output_file = f"./results_{id}.json"
        filename = os.path.join(folder_path, filename)
        program_strs = extract_program_strings(filename)
        executor = ProgramExecutor(
            vocab_size=35,
            max_program_length=50,
            max_execution_steps=100,
            timeout_penalty=-0.1,
            device='cpu'
        )
        karel_envs = {
            'harvester': KarelEnvironment(task='harvester'),
            'cleanHouse': KarelEnvironment(task='cleanHouse'),
            'randomMaze': KarelEnvironment(task='randomMaze')
        }

        results_list = []

        for idx, prog_str in enumerate(program_strs, start=1):
            token_list = tokenize_program(prog_str)
            program_tokens = torch.tensor(token_list, dtype=torch.long)

            program_entry = {
                "program_id": idx,
                "original_program": prog_str,
                "environments": {}
            }

            for env_name, env in karel_envs.items():
                result = executor.execute_single_program(
                    program_tokens,
                    env,
                    return_traces=True
                )

                if 'states' not in result or len(result['states']) == 0:
                    print(f"Warning: no states returned for result {id} (skipping).")
                    continue 
                initial_state = result['states'][0].tolist()

                total_reward = result.get('total_reward')
                success_flag = result.get('success', False)
                error_msg = result.get('error')

                program_entry["environments"][env_name] = {
                    "initial_state": initial_state,
                    "total_reward": total_reward,
                    "success": success_flag,
                    "error": error_msg
                }

            results_list.append(program_entry)

        with open(f"{output_file}", 'w', encoding='utf-8') as f:
            json.dump(results_list, f, ensure_ascii=False, indent=2)

        print(f"results saved to: {output_file}")



if __name__ == '__main__':
    main()
