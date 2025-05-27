#!/usr/bin/env python3
"""
Emergency VAE Diagnosis

The model is generating nonsense despite low loss. This suggests fundamental
issues with training data quality or loss computation.
"""

import torch
import numpy as np
import os
import sys
import h5py
from collections import Counter
from typing import List, Dict

# Add project modules to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dsl.karel_dsl import get_DSL_option_v2
from dsl.tokens import get_vocab_size, get_padding_index

def emergency_data_analysis():
    """Emergency analysis of training data quality"""
    print("🚨 EMERGENCY DATA QUALITY CHECK")
    print("=" * 60)
    
    # Initialize DSL
    dsl = get_DSL_option_v2(seed=42)
    
    # Find dataset
    datadir = "./data/karel_dataset_option_L30_1m_cover_branch"
    if not os.path.exists(datadir):
        print(f"❌ Data directory not found: {datadir}")
        return
    
    print(f"✅ Found data directory: {datadir}")
    
    # Analyze sample programs in detail
    sample_programs = []
    problematic_programs = []
    
    for file_name in os.listdir(datadir):
        if file_name.endswith("hdf5"):
            f_path = os.path.join(datadir, file_name)
            id_file_path = os.path.join(datadir, file_name.replace("data", "id").replace("hdf5", "txt"))
            
            if os.path.exists(id_file_path):
                hdf5_file = h5py.File(f_path, 'r')
                id_file = open(id_file_path, 'r')
                id_list = id_file.readlines()[:100]  # More samples
                
                for program_id in id_list:
                    program_id = program_id.strip().split()[0]
                    program = hdf5_file[program_id]['program'][()]
                    
                    # Convert to string and analyze
                    try:
                        program_str = dsl.intseq2str(program)
                        sample_programs.append(program_str)
                        
                        # Check for problematic patterns
                        if is_problematic_program(program_str):
                            problematic_programs.append(program_str)
                            
                    except Exception as e:
                        print(f"❌ Error converting program {program_id}: {e}")
                        print(f"   Raw tokens: {program[:10]}...")
                
                hdf5_file.close()
                id_file.close()
                break
    
    print(f"\n📊 ANALYSIS RESULTS:")
    print(f"Total programs analyzed: {len(sample_programs)}")
    print(f"Problematic programs: {len(problematic_programs)} ({100*len(problematic_programs)/len(sample_programs):.1f}%)")
    
    # Show examples of each type
    print(f"\n✅ NORMAL PROGRAMS (first 5):")
    normal_programs = [p for p in sample_programs if not is_problematic_program(p)]
    for i, program in enumerate(normal_programs[:5]):
        print(f"  {i+1}: {program}")
    
    print(f"\n❌ PROBLEMATIC PROGRAMS (first 10):")
    for i, program in enumerate(problematic_programs[:10]):
        print(f"  {i+1}: {program}")
        print(f"      Issues: {analyze_program_issues(program)}")
    
    # Token frequency analysis
    all_tokens = []
    for program in sample_programs:
        tokens = program.split()
        all_tokens.extend(tokens)
    
    token_counts = Counter(all_tokens)
    print(f"\n📈 TOP 15 MOST FREQUENT TOKENS:")
    for token, count in token_counts.most_common(15):
        percentage = 100 * count / len(all_tokens)
        print(f"  {token:15s}: {count:4d} ({percentage:5.1f}%)")
    
    # Check for concerning patterns
    ifelse_count = token_counts.get('IFELSE', 0)
    total_tokens = len(all_tokens)
    ifelse_percentage = 100 * ifelse_count / total_tokens
    
    print(f"\n🔍 CRITICAL PATTERN ANALYSIS:")
    print(f"IFELSE frequency: {ifelse_count}/{total_tokens} ({ifelse_percentage:.1f}%)")
    
    if ifelse_percentage > 15:
        print(f"🚨 CRITICAL: IFELSE token extremely frequent ({ifelse_percentage:.1f}%)")
        print(f"   This explains why model generates repetitive IFELSE!")
    
    # Analyze program structure
    structure_analysis = analyze_program_structures(sample_programs)
    print(f"\n🏗️  PROGRAM STRUCTURE ANALYSIS:")
    for issue, count in structure_analysis.items():
        percentage = 100 * count / len(sample_programs)
        print(f"  {issue}: {count}/{len(sample_programs)} ({percentage:.1f}%)")
    
    return sample_programs, problematic_programs, token_counts

def is_problematic_program(program_str: str) -> bool:
    """Check if a program has obvious issues"""
    tokens = program_str.split()
    
    # Check for various issues
    issues = []
    
    # 1. Too many IFELSE tokens
    ifelse_count = tokens.count('IFELSE')
    if ifelse_count > len(tokens) * 0.3:  # More than 30% IFELSE
        issues.append("excessive_ifelse")
    
    # 2. Malformed structure
    if not (tokens[0] == 'DEF' and tokens[1] == 'run'):
        issues.append("bad_start")
    
    if not tokens[-1] == 'm)':
        issues.append("bad_end")
    
    # 3. Unbalanced brackets
    if not check_balanced_brackets(tokens):
        issues.append("unbalanced_brackets")
    
    # 4. Nonsensical sequences
    if has_nonsensical_sequences(tokens):
        issues.append("nonsensical_sequences")
    
    return len(issues) > 0

def analyze_program_issues(program_str: str) -> List[str]:
    """Analyze specific issues with a program"""
    tokens = program_str.split()
    issues = []
    
    # Check IFELSE frequency
    ifelse_count = tokens.count('IFELSE')
    if ifelse_count > 3:
        issues.append(f"too_many_ifelse({ifelse_count})")
    
    # Check structure
    if not (len(tokens) >= 4 and tokens[0] == 'DEF' and tokens[1] == 'run'):
        issues.append("malformed_structure")
    
    # Check for repeated patterns
    if has_repetitive_patterns(tokens):
        issues.append("repetitive_patterns")
    
    return issues

def check_balanced_brackets(tokens: List[str]) -> bool:
    """Check if brackets are balanced"""
    bracket_pairs = {
        'r(': 'r)', 'i(': 'i)', 'e(': 'e)', 'w(': 'w)', 'c(': 'c)', 'm(': 'm)'
    }
    
    stack = []
    for token in tokens:
        if token in bracket_pairs:
            stack.append(bracket_pairs[token])
        elif token in bracket_pairs.values():
            if not stack or stack.pop() != token:
                return False
    
    return len(stack) == 0

def has_nonsensical_sequences(tokens: List[str]) -> bool:
    """Check for nonsensical token sequences"""
    # Look for patterns that don't make sense
    for i in range(len(tokens) - 2):
        # Multiple consecutive IFELSE
        if tokens[i] == 'IFELSE' and tokens[i+1] == 'IFELSE' and tokens[i+2] == 'IFELSE':
            return True
        
        # Numbers without REPEAT
        if tokens[i] in ['1', '2', '3', '4', '5'] and i > 0 and tokens[i-1] != 'REPEAT':
            return True
    
    return False

def has_repetitive_patterns(tokens: List[str]) -> bool:
    """Check for excessive repetition"""
    # Look for tokens repeated more than 3 times consecutively
    for i in range(len(tokens) - 3):
        if tokens[i] == tokens[i+1] == tokens[i+2] == tokens[i+3]:
            return True
    return False

def analyze_program_structures(programs: List[str]) -> Dict[str, int]:
    """Analyze structural issues across all programs"""
    issues = {
        'missing_def_run': 0,
        'missing_m_brackets': 0,
        'excessive_ifelse': 0,
        'unbalanced_brackets': 0,
        'too_short': 0,
        'too_long': 0,
        'nonsensical': 0
    }
    
    for program in programs:
        tokens = program.split()
        
        # Check structure
        if not (len(tokens) >= 4 and tokens[0] == 'DEF' and tokens[1] == 'run'):
            issues['missing_def_run'] += 1
        
        if not (tokens[2] == 'm(' and tokens[-1] == 'm)'):
            issues['missing_m_brackets'] += 1
        
        # Check IFELSE frequency
        ifelse_count = tokens.count('IFELSE')
        if ifelse_count > len(tokens) * 0.2:
            issues['excessive_ifelse'] += 1
        
        # Check length
        if len(tokens) < 5:
            issues['too_short'] += 1
        elif len(tokens) > 20:
            issues['too_long'] += 1
        
        # Check brackets
        if not check_balanced_brackets(tokens):
            issues['unbalanced_brackets'] += 1
        
        # Check for nonsense
        if has_nonsensical_sequences(tokens):
            issues['nonsensical'] += 1
    
    return issues

def recommend_fixes(token_counts, structure_issues, problematic_percentage):
    """Recommend specific fixes based on analysis"""
    print(f"\n🛠️  RECOMMENDED FIXES:")
    print("=" * 60)
    
    if problematic_percentage > 30:
        print(f"🚨 CRITICAL: {problematic_percentage:.1f}% of training data is problematic!")
        print(f"   SOLUTION: Filter training data before training")
    
    ifelse_percentage = 100 * token_counts.get('IFELSE', 0) / sum(token_counts.values())
    if ifelse_percentage > 10:
        print(f"🚨 CRITICAL: IFELSE token is {ifelse_percentage:.1f}% of all tokens!")
        print(f"   SOLUTION: This explains repetitive IFELSE in output")
        print(f"   FIX: Clean training data or add token frequency balancing")
    
    print(f"\n📋 IMMEDIATE ACTIONS:")
    print(f"1. STOP current training - model learning from bad data")
    print(f"2. FILTER training data - remove problematic programs")
    print(f"3. CHECK data preprocessing pipeline")
    print(f"4. RETRAIN with clean data")

def main():
    """Run emergency diagnosis"""
    print("🚨 EMERGENCY VAE DIAGNOSIS")
    print("=" * 80)
    
    sample_programs, problematic_programs, token_counts = emergency_data_analysis()
    
    if sample_programs:
        problematic_percentage = 100 * len(problematic_programs) / len(sample_programs)
        structure_issues = analyze_program_structures(sample_programs)
        recommend_fixes(token_counts, structure_issues, problematic_percentage)
    
    print("\n" + "=" * 80)
    print("DIAGNOSIS COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()