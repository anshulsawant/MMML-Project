import json
import re
import argparse
from datasets import load_dataset
from tqdm import tqdm

def extract_answer(solution_text: str) -> str:
    if not solution_text:
        return ""
    
    answer_match = re.search(r'<answer>(.*?)</answer>', solution_text, re.DOTALL)
    if answer_match:
        block = answer_match.group(1)
    else:
        block = solution_text

    # The baseline models occasionally append </think> or random explanations on the very same line.
    # We must explicitly slice using the first HTML tag to avoid capturing arbitrary trailing text
    final_ans_matches = re.findall(r'Final\s+[Aa]nswer[^\d\w\$\(\\\-]*\s*([^\n]*)', block, re.IGNORECASE)
    if final_ans_matches:
        ans_str = final_ans_matches[-1].strip()
        ans_str = ans_str.split('</')[0]
    else:
        bold_match = re.findall(r'\\(?:boldsymbol|mathbf)\{([^}]+)\}', block)
        if bold_match:
            ans_str = bold_match[-1].strip()
            ans_str = ans_str.split('</')[0]
        else:
            lines = [l.strip() for l in block.split('\n') if l.strip()]
            ans_str = lines[-1] if lines else ""
            ans_str = ans_str.split('</')[0]

    # 1. Strip styling wrappers but leave their closing braces for now
    ans_str = re.sub(r'\\(?:boldsymbol|mathbf|mathrm|text|mathit|boxed|fbox)\{', '', ans_str)
    
    # 2. Handle inner expressions like \sqrt{a} -> sqrt(a)
    ans_str = re.sub(r'\\sqrt\{([^}]+)\}', r'sqrt(\1)', ans_str)
    ans_str = re.sub(r'\\sqrt\s*([^ ]+)', r'sqrt(\1)', ans_str)
    
    # 3. Handle fractions like \frac{a}{b} -> a/b
    ans_str = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'\1/\2', ans_str)
    ans_str = re.sub(r'\\frac\{?([0-9]+)\}?', r'\1', ans_str)
    
    ans_str = re.sub(r'\\[\(|\)]', '', ans_str)
    ans_str = re.sub(r'\$', '', ans_str)
    ans_str = ans_str.replace('^\\circ', '').replace('\\circ', '')
    ans_str = re.sub(r'degrees?', '', ans_str, flags=re.IGNORECASE)
    ans_str = ans_str.replace('\\pi', 'pi').replace('π', 'pi')
    
    # Chinese prefix strip "最终答案："
    ans_str = re.sub(r'最终答案：\s*', '', ans_str)
    
    # Strip leading English transition phrases
    ans_str = re.sub(r'^(?:Thus|So|Therefore|Hence|Then|The answer is|Final answer|Answer|value is)\s*[,:]?\s*', '', ans_str, flags=re.IGNORECASE)
    
    # If the string has an equals sign or an arrow (e.g. "The minimum length is EF = 2.4"), just take the final value
    if '=' in ans_str:
        ans_str = ans_str.split('=')[-1]
    if '→' in ans_str:
        ans_str = ans_str.split('→')[-1]
    if '\\rightarrow' in ans_str:
        ans_str = ans_str.split('\\rightarrow')[-1]
    if '\\to' in ans_str:
        ans_str = ans_str.split('\\to')[-1]
        
    # Strip trailing approx words block WITHOUT eating preceding mathematical parentheses
    ans_str = re.sub(r'\s*\(\s*(?:approximately|approx|or|but|rounded|if|equivalent|about)\b[^\)]*\)', '', ans_str, flags=re.IGNORECASE)
    ans_str = re.sub(r'\s*approx.*$', '', ans_str, flags=re.IGNORECASE)
    ans_str = re.sub(r'\s*or\s+.*$', '', ans_str, flags=re.IGNORECASE)
    
    # Pre-parse "The radius of the circle is 5.0" style sentences before we strip they drop
    if ' is ' in ans_str:
        tokens = ans_str.split(' is ')
        # Snip anything under 30 chars containing no algebraic separators as a safe prefix.
        # This catches "The exact circumference of the circle is 8pi" -> "8pi"
        if len(tokens[-1]) < 30 and '=' not in tokens[-1]:
            ans_str = tokens[-1].strip()
            
    # Now that logic is done, strip all remaining stray brackets from the stripped wrappers
    ans_str = ans_str.replace('{', '').replace('}', '').replace('**', '')
    
    # Clean up weird trailing math punctuation
    ans_str = re.sub(r'[,;:\.]+$', '', ans_str)
    ans_str = ans_str.strip(' .*')
    
    # Nuke trailing "Since the problem doesn't specify" type sentences ONLY if they are detached sentences
    ans_str = re.sub(r'\s+[A-Z][A-Za-z]{3,}.*$', '', ans_str)
    
    # Strip common units
    ans_str = re.sub(r'\s*(?:feet|foot|inches|inch|centimeters|centimeter|cm|mm|meters|meter|units|unit|square)\b', '', ans_str, flags=re.IGNORECASE)
    
    # If the answer is literally just English articles or punctuation, wipe it.
    if ans_str.lower() in ['the', 'a', 'an', '[', ']', '(', ')', 'this', 'that', 'r', 'is', 'are', 'n + m - 180']:
        ans_str = ""
    # Strip standalone "sqrt " or trailing "sqrt )" or "or" that parsed poorly
    if ans_str.endswith('sqrt') or ans_str.endswith('sqrt )'):
        ans_str = ans_str.replace('sqrt )', '').replace('sqrt', '').strip()
    ans_str = ans_str.replace(' or', '')
    
    # Clean up trailing unclosed parentheticals like "(square" or "(exact" or ": 84.85"
    ans_str = re.sub(r'\s*\([a-z].*$', '', ans_str, flags=re.IGNORECASE)
    ans_str = re.sub(r'^\s*:\s*', '', ans_str)
    ans_str = ans_str.replace('fracsqrt', 'sqrt')
    
    # Strip any trailing English explanations: e.g "20 (since side is...)"
    # BUT only if it looks like a full language sentence (3+ letter word) to avoid breaking "4 + pi"
    if re.search(r'[A-Za-z]{3,} ', ans_str) and 'sqrt' not in ans_str and 'pi' not in ans_str:
        terms = ans_str.split()
        if terms:
            ans_str = terms[-1]
    
    # Remove any lingering \
    ans_str = ans_str.replace('\\', '')
    
    return ans_str

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=str, default="data/ground_truths.json")
    args = parser.parse_args()
    
    print("Loading dataset...")
    dataset = load_dataset('parquet', data_files='GeoThought/playground/data/geo_thought/Geo-Thought-6K.parquet', split='train')
    
    ground_truths = {}
    
    num_items = len(dataset) if args.limit is None else min(args.limit, len(dataset))
    
    for idx in tqdm(range(num_items), desc="Extracting Ground Truths"):
        item = dataset[idx]
        sol = item.get('solution', '')
        ans_str = extract_answer(sol)
        
        img_filename = f"GeoThought/playground/data/images/samples/sample_{idx}.jpg"
        
        ground_truths[img_filename] = ans_str
        
    with open(args.output, 'w') as f:
        json.dump(ground_truths, f, indent=2)
        
    print(f"\nWrote {len(ground_truths)} answers to {args.output}")

    # Print a few random extractions
    import random
    print("\n--- Random Samples ---")
    keys = list(ground_truths.keys())
    for k in random.sample(keys, min(10, len(keys))):
        print(f"{k}: {ground_truths[k]}")

if __name__ == "__main__":
    main()
