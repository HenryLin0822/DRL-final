import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.utils.rnn as rnn
import numpy as np
import random
import os
from tqdm import tqdm
import h5py
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import config
from dsl.karel_dsl import get_DSL_option_v2
class ProgramDataset(Dataset):
    """Karel programs dataset."""

    def __init__(self, program_list, config, num_program_tokens, num_agent_actions, device):
        """ Init function for karel program dataset

        Parameters:
            :param program_list (list): list containing information about each program in dataset
            :param config (dict): all configs in dict format
            :param num_program_tokens (int): number of program tokens in karel DSL
            :param num_agent_actions (int): number of actions karel agent can take
            :param device(torch.device): dataset target device: torch.device('cpu') or torch.device('cuda:X')

        Returns: None
        """
        self.device = device
        self.config = config
        self.programs = program_list
        # need this +1 as DEF token is input to decoder, loss will be calculated only from run token
        self.max_program_len = config['dsl']['max_program_len'] + 1
        self.num_program_tokens = num_program_tokens
        self.num_agent_actions = num_agent_actions

    def _dsl_to_prl(self, program_seq):
        """ DSL tokens to PRL tokens mapping.
        PRL tokens refer to a shorter list of karel program tokens, which can be specified through mapping_karel2prl.txt

        Parameters:
            :param program_seq (list): program as a sequence of integers

        Returns: list
            :return: new program with PRL token mapping
        """
        def func(x):
            return self.config['prl_tokens'].index(self.config['dsl2prl_mapping'][self.config['dsl_tokens'][x]])
        return np.array(list(map(func, program_seq)), program_seq.dtype)

    def __len__(self):
        return len(self.programs)

    def __getitem__(self, idx):
        program_id, sample, exec_data = self.programs[idx]
        sample = self._dsl_to_prl(sample) if self.config['use_simplified_dsl'] else sample

        sample = torch.from_numpy(sample).to(self.device).to(torch.long)
        program_len = sample.shape[0]
        sample_filler = torch.tensor((self.max_program_len - program_len) * [self.num_program_tokens - 1],
                                     device=self.device, dtype=torch.long)
        sample = torch.cat((sample, sample_filler))

        mask = torch.zeros((self.max_program_len, 1), device=self.device, dtype=torch.bool)
        mask[:program_len] = 1

        # load exec data
        s_h, a_h, a_h_len = exec_data
        s_h = torch.tensor(s_h, device=self.device, dtype=torch.float32)
        a_h = torch.tensor(a_h, device=self.device, dtype=torch.int16)
        a_h_len = torch.tensor(a_h_len, device=self.device, dtype=torch.int16)

        packed_a_h = rnn.pack_padded_sequence(a_h, a_h_len.to("cpu"), batch_first=True, enforce_sorted=False)
        padded_a_h, a_h_len = rnn.pad_packed_sequence(packed_a_h, batch_first=True,
                                                      padding_value=self.num_agent_actions-1,
                                                      total_length=self.config['max_demo_length'] - 1)

        return sample, program_id, mask, s_h, padded_a_h, a_h_len.to(self.device)
    
def get_exec_data(hdf5_file, program_id, num_agent_actions):
    def func(x):
        s_h, s_h_len = x
        assert s_h_len > 1
        return np.expand_dims(s_h[0], 0)

    s_h = np.moveaxis(np.copy(hdf5_file[program_id]['s_h']), [-1,-2,-3], [-3,-1,-2])
    a_h = np.copy(hdf5_file[program_id]['a_h'])
    s_h_len = np.copy(hdf5_file[program_id]['s_h_len'])
    a_h_len = np.copy(hdf5_file[program_id]['a_h_len'])

    # expand demo length if max_demo_len==1
    if s_h.shape[1] == 1:
        s_h = np.concatenate((np.copy(s_h), np.copy(s_h)), axis=1)
        #print("a_h shape: ", a_h.shape)
        a_h = np.ones((s_h.shape[0], 1))

    # Add no-op actions for empty demonstrations
    for i in range(s_h_len.shape[0]):
        if a_h_len[i] == 0:
            assert s_h_len[i] == 1
            a_h_len[i] += 1
            s_h_len[i] += 1
            s_h[i][1, :, :, :] = s_h[i][0, :, :, :]
            a_h[i][0] = num_agent_actions - 1

    # select input state from demonstration executions
    results = map(func, zip(s_h, s_h_len))

    s_h = np.stack(list(results))
    return s_h, a_h, a_h_len

def make_datasets(datadir, config, num_program_tokens, num_agent_actions, device, dsl):
    """ Given the path to main dataset, split the data into train, valid, test and create respective pytorch Datasets

    Parameters:
        :param datadir (str): patth to main dataset (should contain 'data.hdf5' and 'id.txt')
        :param config (dict):  all configs in dict format
        :param num_program_tokens (int): number of program tokens in karel DSL
        :param num_agent_actions (int): number of actions karel agent can take
        :param device(torch.device): dataset target device: torch.device('cpu') or torch.device('cuda:X')

    Returns:
        :return train_dataset(torch.utils.data.Dataset): training dataset
        :return valid_dataset(torch.utils.data.Dataset): validation dataset
        :return test_dataset(torch.utils.data.Dataset): test dataset

    """
    program_list = []
    r_eq_program_count = 0
    drop_program_count = 0
    seen_programs = set()


    print("Loading data from: ", datadir)
    for file_name in tqdm(os.listdir(datadir)):
        if file_name.endswith("hdf5"):
            f_path = os.path.join(datadir, file_name)
            id_file_path = os.path.join(datadir, file_name.replace("data", "id").replace("hdf5", "txt"))
            


            hdf5_file = h5py.File(f_path, 'r')
            id_file = open(id_file_path, 'r')
            id_list = id_file.readlines()
            for program_id in id_list:
                program_id = program_id.strip().split()[0]
                program = hdf5_file[program_id]['program'][()]
                valid_flag = True 
                
                random_code_str = dsl.intseq2str(program)
                
                if random_code_str in seen_programs:
                    continue

                if program.shape[0] < config['dsl']['max_program_len'] and valid_flag:
                    exec_data = get_exec_data(hdf5_file, program_id, num_agent_actions)
                    program_list.append((program_id, program, exec_data))
                    seen_programs.add(random_code_str)


            hdf5_file.close()
            id_file.close()



    random.shuffle(program_list)

    train_r, val_r, test_r = 0.85, 0.15, 0.0
    split_idx1 = int(train_r*len(program_list))
    split_idx2 = int((train_r+val_r)*len(program_list))
    train_program_list = program_list[:split_idx1]
    valid_program_list = program_list[split_idx1:split_idx2]
    test_program_list = valid_program_list #program_list[split_idx2:]

    train_dataset = ProgramDataset(train_program_list, config, num_program_tokens, num_agent_actions, device)
    val_dataset = ProgramDataset(valid_program_list, config, num_program_tokens, num_agent_actions, device)
    test_dataset = ProgramDataset(test_program_list, config, num_program_tokens, num_agent_actions, device)
    return train_dataset, val_dataset, test_dataset


if __name__ == "__main__":
        datadir = "../data/karel_dataset_option_L30_1m_cover_branch"
        config = config
        dsl = get_DSL_option_v2(seed=config['seed'], environment=config['rl']['envs']['executable']['name'])
        config['dsl']['num_agent_actions'] = len(dsl.action_functions) + 1  
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_program_tokens = 35
        p_train_dataset, p_val_dataset, p_test_dataset = make_datasets(datadir, config,
                                                                    num_program_tokens,
                                                                    config['dsl']['num_agent_actions'], device,
                                                                    dsl)
        print(len(p_train_dataset))
        print(len(p_val_dataset))
        print(len(p_test_dataset))