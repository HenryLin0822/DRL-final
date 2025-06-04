from __future__ import annotations
import math
import torch
from typing import Dict, List
# switch to Llama model via LangChain
import numpy as np
# for downloading model

from llm.prompt_generator import PromptGenerator
from llm.utils import (
    get_program_str_from_llm_response_dsl,
    get_program_str_from_llm_response_python,
)
from prog_policies.utils import get_env_name
from prog_policies.base import BaseDSL, dsl_nodes
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

class LLMProgramGenerator:
    def __init__(
        self,
        seed: int,
        task: str,
        dsl: BaseDSL,
        llm_program_num: int,
        temperature: float = 1.0,
        top_p: float = 1.0,
        action_shots: int = 0,
        perception_shots: int = 0,
        program_shots: int = 0,
    ) -> None:
        self.seed = seed
        self.ratio = 1.5
        self.task = task
        self.env_name = get_env_name(task)
        self.dsl = dsl
        self.llm_program_num = llm_program_num
        self.temperature = temperature
        self.top_p = top_p

        # Model ID
        model_name = "Qwen/Qwen3-14B"

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.add_special_tokens({"pad_token": self.tokenizer.eos_token})

        # Quantization config (4-bit)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        # Load model with quantization
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            trust_remote_code=True,
            device_map="auto",
            low_cpu_mem_usage=True,
        )

        # Apply LoRA
        peft_config = LoraConfig(
            r=64,
            lora_alpha=128,
            lora_dropout=0.2,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "up_proj", "down_proj", "gate_proj",
                "k_proj", "q_proj", "v_proj", "o_proj",
            ],
        )
        self.model = get_peft_model(base_model, peft_config)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.eval()

        # RNG and prompts
        self.np_rng = np.random.RandomState(self.seed)
        self.action_shots = action_shots
        self.perception_shots = perception_shots
        self.program_shots = program_shots
        self.prompt_generator = PromptGenerator(
            self.task,
            self.action_shots,
            self.perception_shots,
            self.program_shots,
        )

    def _call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        n: int,
    ) -> list[str]:
        inputs = self.tokenizer(
            f"{system_prompt}{self.tokenizer.eos_token}{user_prompt}",
            return_tensors="pt",
        ).to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=self.temperature,
            top_p=self.top_p,
            num_return_sequences=n,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        return [
            self.tokenizer.decode(o, skip_special_tokens=True).strip()
            for o in outputs
        ]

    def _get_program_list_from_llm_response_python_to_dsl(self, response) -> list[str]:
        program_str_list = []
        for x in response:
            tmp = []
            try:
                program_str = get_program_str_from_llm_response_python(x, env_name=self.env_name)
                tmp.append(program_str)
            except:
                pass
            
            try:
                program_str = get_program_str_from_llm_response_dsl(x, env_name=self.env_name)
                tmp.append(program_str)
            except:
                pass
            
            program_str_list.append(tmp)
        return program_str_list
    
    
    def _get_program_list_from_llm_response_python(self, response) -> list[str]:
        program_str_list = []
        for x in response:
            try:
                program_str = get_program_str_from_llm_response_python(x, env_name=self.env_name)
                program_str_list.append(program_str)
            except:
                pass
        return program_str_list
    
    def _get_program_list_from_llm_response_dsl(self, response) -> list[str]:
        program_str_list = []
        for x in response:
            try:
                program_str = get_program_str_from_llm_response_dsl(x, env_name=self.env_name)
                program_str_list.append(program_str)
            except:
                pass
        return program_str_list

    def get_program_list_python_to_dsl(self) -> tuple[list, dict]:
        program_list = []
        record_list = []
        attempts = 0
        program_num = self.llm_program_num
        while len(program_list) < program_num:
            attempts += 1
            seed = self.np_rng.randint(0, 2**32)
            llm_program_num = math.ceil((program_num - len(program_list)) * self.ratio)
            system_prompt = self.prompt_generator.get_system_prompt_python_to_dsl()
            user_prompt = self.prompt_generator.get_user_prompt_python_to_dsl()
            llm_response = self._call_llm(system_prompt, user_prompt, llm_program_num)
            program_str_list = self._get_program_list_from_llm_response_python_to_dsl(llm_response)
            for candidates in program_str_list:
                tmp = []
                for candidate in candidates:
                    print("===" * 20, flush=True)
                    print(candidate, flush=True)
                    print("===" * 20, flush=True)
                    try:
                        program = self.dsl.parse_str_to_node(candidate)
                        tmp.append(program)
                    except:
                        pass
                if len(tmp) > 0:
                    program_list.append(self.np_rng.choice(tmp))
                    
            available_program_num = len(program_list)
            print(f"Attempts: {attempts}, Program_nums: {available_program_num}")
            
            if len(program_list) > program_num:
                program_list = program_list[:program_num]
            program_str_list = [self.dsl.parse_node_to_str(program) for program in program_list]
            
            record_list.append(
                {
                    "seed": seed,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "llm_response": llm_response,
                    "available_program_num": available_program_num,
                    "program_str_list": program_str_list,
                }
            )

        log = {"attemps": attempts, "record_list": record_list}

        return program_list, log
    
    def get_program_list_python(self) -> tuple[list, dict]:
        program_list = []
        record_list = []
        attempts = 0
        program_num = self.llm_program_num
        while len(program_list) < program_num:
            attempts += 1
            seed = self.np_rng.randint(0, 2**32)
            llm_program_num = math.ceil((program_num - len(program_list)) * self.ratio)
            system_prompt = self.prompt_generator.get_system_prompt_python()
            user_prompt = self.prompt_generator.get_user_prompt_python()
            llm_response = self._call_llm(system_prompt, user_prompt, llm_program_num)
            program_str_list = self._get_program_list_from_llm_response_python(llm_response)
            for program_str in program_str_list:
                print("===" * 20, flush=True)
                print(program_str, flush=True)
                print("===" * 20, flush=True)
                try:
                    
                    program = self.dsl.parse_str_to_node(program_str)
                    program_list.append(program)
                except:
                    pass
                
            available_program_num = len(program_list)
            print(f"Attempts: {attempts}, Program_nums: {available_program_num}")
            
            if len(program_list) > program_num:
                program_list = program_list[:program_num]
            program_str_list = [self.dsl.parse_node_to_str(program) for program in program_list]
            
            record_list.append(
                {
                    "seed": seed,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "llm_response": llm_response,
                    "available_program_num": available_program_num,
                    "program_str_list": program_str_list,
                }
            )

        log = {"attemps": attempts, "record_list": record_list}

        return program_list, log
    
    def get_program_list_dsl(self) -> tuple[list, dict]:
        program_list = []
        record_list = []
        attempts = 0
        program_num = self.llm_program_num
        while len(program_list) < program_num:
            attempts += 1
            seed = self.np_rng.randint(0, 2**32)
            llm_program_num = math.ceil((program_num - len(program_list)) * self.ratio)
            system_prompt = self.prompt_generator.get_system_prompt_dsl()
            user_prompt = self.prompt_generator.get_user_prompt_dsl()
            print(system_prompt, flush=True)
            print(user_prompt, flush=True)
            llm_response = self._call_llm(system_prompt, user_prompt, llm_program_num)
            program_str_list = self._get_program_list_from_llm_response_dsl(llm_response)
            for program_str in program_str_list:
                print("===" * 20, flush=True)
                print(program_str, flush=True)
                print("===" * 20, flush=True)
                try:
                    program = self.dsl.parse_str_to_node(program_str)
                    program_list.append(program)
                except:
                    pass
            
            available_program_num = len(program_list)
            print(f"Attempts: {attempts}, Program_nums: {available_program_num}")
            
            if len(program_list) > program_num:
                program_list = program_list[:program_num]
            program_str_list = [self.dsl.parse_node_to_str(program) for program in program_list]    
            
            record_list.append(
                {
                    "seed": seed,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "llm_response": llm_response,
                    "available_program_num": available_program_num,
                    "program_str_list": program_str_list,
                }
            )

        log = {"attemps": attempts, "record_list": record_list}

        return program_list, log

    def get_program_list_revision_regeneration_with_reward(
        self,
        progs_rewards: list[dsl_nodes.Program, float]
    ):
        program_list = []
        record_list = []
        attempts = 0
        program_num = self.llm_program_num
        while len(program_list) < program_num:
            attempts += 1
            seed = self.np_rng.randint(0, 2**32)
            llm_program_num = math.ceil((program_num - len(program_list)) * self.ratio)
            system_prompt = self.prompt_generator.get_system_prompt_python_to_dsl()
            user_prompt = self.prompt_generator.get_user_prompt_revision_regeneration_with_reward(progs_rewards, self.dsl)
            llm_response = self._call_llm(system_prompt, user_prompt, llm_program_num)
            program_str_list = self._get_program_list_from_llm_response_python_to_dsl(llm_response)
            for candidates in program_str_list:
                tmp = []
                for candidate in candidates:
                    try:
                        program = self.dsl.parse_str_to_node(candidate)
                        tmp.append(program)
                    except:
                        pass
                if len(tmp) > 0:
                    program_list.append(self.np_rng.choice(tmp))
            
            available_program_num = len(program_list)
            print(f"Attempts: {attempts}, Program_nums: {available_program_num}")
            
            if len(program_list) > program_num:
                program_list = program_list[:program_num]
            program_str_list = [self.dsl.parse_node_to_str(program) for program in program_list]  
            
            record_list.append(
                {
                    "seed": seed,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "llm_response": llm_response,
                    "available_program_num": available_program_num,
                    "program_str_list": program_str_list,
                }
            )

        log = {"attemps": attempts, "record_list": record_list}

        return program_list, log

    def get_program_list_revision_regeneration(
        self,
        previous_program_list: List[dsl_nodes.Program],
    ):
        program_list = []
        record_list = []
        attempts = 0
        program_num = self.llm_program_num
        while len(program_list) < program_num:
            attempts += 1
            seed = self.np_rng.randint(0, 2**32)
            llm_program_num = math.ceil((program_num - len(program_list)) * self.ratio)
            system_prompt = self.prompt_generator.get_system_prompt_python_to_dsl()
            user_prompt = self.prompt_generator.get_user_prompt_revision_regeneration(previous_program_list, self.dsl)
            llm_response = self._call_llm(system_prompt, user_prompt, llm_program_num)
            program_str_list = self._get_program_list_from_llm_response_python_to_dsl(llm_response)
            for candidates in program_str_list:
                tmp = []
                for candidate in candidates:
                    try:
                        program = self.dsl.parse_str_to_node(candidate)
                        tmp.append(program)
                    except:
                        pass
                if len(tmp) > 0:
                    program_list.append(self.np_rng.choice(tmp))
            
            available_program_num = len(program_list)
            print(f"Attempts: {attempts}, Program_nums: {available_program_num}")
            
            if len(program_list) > program_num:
                program_list = program_list[:program_num]
            program_str_list = [self.dsl.parse_node_to_str(program) for program in program_list]  
            
            record_list.append(
                {
                    "seed": seed,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "llm_response": llm_response,
                    "available_program_num": available_program_num,
                    "program_str_list": program_str_list,
                }
            )

        log = {"attemps": attempts, "record_list": record_list}

        return program_list, log
    
    def get_program_list_revision_agent_execution_trace(
        self,
        reward: float, logs: list[dict[str, str]], average_reward: float,
    ) -> tuple[list[dsl_nodes.Program], dict]:
        program_list = []
        record_list = []
        attempts = 0
        program_num = self.llm_program_num
        while len(program_list) < program_num:
            attempts += 1
            seed = self.np_rng.randint(0, 2**32)
            llm_program_num = math.ceil((program_num - len(program_list)) * self.ratio)
            system_prompt = self.prompt_generator.get_system_prompt_python_to_dsl()
            user_prompt = self.prompt_generator.get_user_prompt_revision_agent_execution_trace(reward, logs, average_reward)
            llm_response = self._call_llm(system_prompt, user_prompt, llm_program_num)
            program_str_list = self._get_program_list_from_llm_response_python_to_dsl(llm_response)
            for candidates in program_str_list:
                tmp = []
                for candidate in candidates:
                    try:
                        program = self.dsl.parse_str_to_node(candidate)
                        tmp.append(program)
                    except:
                        pass
                if len(tmp) > 0:
                    program_list.append(self.np_rng.choice(tmp))
            
            available_program_num = len(program_list)
            print(f"Attempts: {attempts}, Program_nums: {available_program_num}")
            
            if len(program_list) > program_num:
                program_list = program_list[:program_num]
            program_str_list = [self.dsl.parse_node_to_str(program) for program in program_list]  
            
            record_list.append(
                {
                    "seed": seed,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "llm_response": llm_response,
                    "available_program_num": available_program_num,
                    "program_str_list": program_str_list,
                }
            )

        log = {"attemps": attempts, "record_list": record_list}

        return program_list, log
    
    def get_program_list_revision_agent_program_execution_trace(
        self,
        reward: float, logs: list[dict[str, str]], average_reward: float,
    ) -> tuple[list[dsl_nodes.Program], dict]:
        program_list = []
        record_list = []
        attempts = 0
        program_num = self.llm_program_num
        while len(program_list) < program_num:
            attempts += 1
            seed = self.np_rng.randint(0, 2**32)
            llm_program_num = math.ceil((program_num - len(program_list)) * self.ratio)
            system_prompt = self.prompt_generator.get_system_prompt_python_to_dsl()
            user_prompt = self.prompt_generator.get_user_prompt_revision_agent_program_execution_trace(reward, logs, average_reward)
            llm_response = self._call_llm(system_prompt, user_prompt, llm_program_num)
            program_str_list = self._get_program_list_from_llm_response_python_to_dsl(llm_response)
            for candidates in program_str_list:
                tmp = []
                for candidate in candidates:
                    try:
                        program = self.dsl.parse_str_to_node(candidate)
                        tmp.append(program)
                    except:
                        pass
                if len(tmp) > 0:
                    program_list.append(self.np_rng.choice(tmp))
            
            available_program_num = len(program_list)
            print(f"Attempts: {attempts}, Program_nums: {available_program_num}")
            
            if len(program_list) > program_num:
                program_list = program_list[:program_num]
            program_str_list = [self.dsl.parse_node_to_str(program) for program in program_list]  
            
            record_list.append(
                {
                    "seed": seed,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "llm_response": llm_response,
                    "available_program_num": available_program_num,
                    "program_str_list": program_str_list,
                }
            )
        log = {"attemps": attempts, "record_list": record_list}
        return program_list, log
    
