import numpy as np
import torch
import h5py

from environments.karel_env import KarelEnvironment
from models.program_executor import ProgramExecutor

# ------------------------------------------------------------------------------
# 1. Define the token map and its reverse for lookup
# ------------------------------------------------------------------------------
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

# Build reverse map: token string → integer ID
reverse_token_map = {tok: idx for idx, tok in token_map.items()}

# ------------------------------------------------------------------------------
# 2. Specify your TASK name (used in filenames) and paths
# ------------------------------------------------------------------------------
TASK = "harvester"

input_filename = f"dsl_programs_{TASK}.txt"
hdf5_filename = f"results_{TASK}.hdf5"
txt_output_filename = f"results_{TASK}.txt"

# ------------------------------------------------------------------------------
# 3. Prepare executor and environment once (reuse for all runs)
# ------------------------------------------------------------------------------
executor = ProgramExecutor(
    vocab_size=35,
    max_program_length=500,
    max_execution_steps=200,
    timeout_penalty=-0.1,
    device='cpu'
)
env = KarelEnvironment(task=TASK)

# ------------------------------------------------------------------------------
# 4. Read the DSL‐programs TXT file, tokenize, run each program 10 times, and save
# ------------------------------------------------------------------------------
with open(input_filename, 'r', encoding='utf-8') as infile, \
     h5py.File(hdf5_filename, 'w') as hdf5_file, \
     open(txt_output_filename, 'w', encoding='utf-8') as txt_out:
    
    print("0601 version 1")
    for line in infile:
        line = line.strip()
        if not line:
            continue

        # Parse "no_{id}" and the program string
        parts = line.split(maxsplit=1)
        id_str = parts[0]                   # e.g. "no_1"
        program_str = parts[1] if len(parts) > 1 else ""

        # Tokenize using reverse_token_map
        token_strings = program_str.split()
        try:
            token_ids = np.array(
                [reverse_token_map[tok] for tok in token_strings],
                dtype=np.int8
            )
        except KeyError as e:
            raise ValueError(f"Token '{e.args[0]}' not found in token_map.") from None

        # Convert to a PyTorch tensor for executor
        program_tokens = torch.tensor(token_ids, dtype=torch.long)

        # Prepare lists to collect 10 demos
        all_a_h = []
        all_a_h_len = []
        all_s_h = []
        all_s_h_len = []

        # Run the same program 10 times
        for _ in range(10):
            result = executor.execute_single_program(
                program_tokens,
                env,
                return_traces=True
            )


            a_h = np.array(result['actions'], dtype=np.int8)       # shape (max_a_h_len,)
            a_h_len = np.int16(result['action_length'])              # scalar
            s_h = np.array(result['states'], dtype=bool)          # shape (max_s_h_len, H, W, C)
            s_h_len = np.int16(np.array(result['states']).shape[0])              # scalar

            all_a_h.append(a_h)
            all_a_h_len.append(a_h_len)
            all_s_h.append(s_h)
            all_s_h_len.append(s_h_len)

        # Stack along a new first dimension (10 demos)
        # Assumes that for this program, each execution returned the same array shapes:
        #   - a_h.shape == (max_a_h_len,)
        #   - s_h.shape == (max_s_h_len, H, W, C)
        max_a_h_len = max(all_a_h_len)
        padded_a_h = [
            np.pad(a_h, (0, max_a_h_len - a_h.shape[0]), constant_values=0)
            for a_h in all_a_h
        ]
        a_h_array = np.stack(padded_a_h, axis=0)        # → shape (10, max_a_h_len)
        a_h_len_array = np.stack(all_a_h_len, axis=0) # → shape (10,)

        max_s_h_len = max(all_s_h_len)
        padded_s_h = [
            np.pad(
                s_h,
                ((0, max_s_h_len - s_h.shape[0]),   # pad on the “time” axis
                (0, 0),                            # no pad on H axis
                (0, 0),                            # no pad on W axis
                (0, 0)),                           # no pad on C axis
                constant_values=False               # pad- value is False (i.e., zeros)
            )
            for s_h in all_s_h
        ]
        s_h_array = np.stack(padded_s_h, axis=0)         # → shape (10, max_s_h_len, H, W, C)
        s_h_len_array = np.stack(all_s_h_len, axis=0) # → shape (10,)

        # Compute group_name using program length and max_s_h_len
        prog_len = len(token_ids)
        group_name = f"{id_str}_prog_len_{prog_len}_max_s_h_len_{max_s_h_len}"

        # Save into HDF5 under this group
        grp = hdf5_file.create_group(group_name)
        grp.create_dataset('a_h', data=a_h_array)           # (10, max_a_h_len)
        grp.create_dataset('a_h_len', data=a_h_len_array)   # (10,)
        grp.create_dataset('s_h', data=s_h_array)           # (10, max_s_h_len, H, W, C)
        grp.create_dataset('s_h_len', data=s_h_len_array)   # (10,)
        grp.create_dataset('program', data=token_ids)       # (prog_len,)

        # Also write a line in the TXT summary
        txt_out.write(f"{group_name} {program_str}\n")

print(f"Finished. HDF5 results saved to '{hdf5_filename}', textual summary saved to '{txt_output_filename}'.")
