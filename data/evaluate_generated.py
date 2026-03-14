import json
import re
import math

def clean_gen_ans(text):
    if 'Step 4 [Final Conclusion]:' in text:
        text = text.split('Step 4 [Final Conclusion]:')[-1]
    
    # Handle weird lowercase markdown leaks
    for delimiter in ['### step 4', 'step 4 [final conclusion]:', 'final conclusion:', '###step4']:
        if delimiter in text.lower():
            text = re.split(re.escape(delimiter), text, flags=re.IGNORECASE)[-1]

    ans = text.strip()
    
    # If the model blasted out multiple paragraphs inside the conclusion block, grab the very last line
    lines = [l.strip() for l in ans.split('\n') if l.strip()]
    if lines:
        ans = lines[-1]
    
    # Strip basic math fluff
    ans = re.sub(r'\\(?:boxed|boldsymbol|mathbf|mathrm|text|mathit|fbox)\{', '', ans)
    
    # Strip phrasing
    ans = re.sub(r'^(?:Thus|So|Therefore|Hence|Then|The answer|Answer)\b.*?\bis\s+', '', ans, flags=re.IGNORECASE)
    ans = re.sub(r'^(?:Any value greater than )', '', ans, flags=re.IGNORECASE)
    
    if '=' in ans:
        ans = ans.split('=')[-1]
    if 'is' in ans.lower() and len(ans) < 30:
        ans = re.split(r'\bis\b', ans, flags=re.IGNORECASE)[-1]
        
    ans = ans.replace('^\\circ', '').replace('\\circ', '')
    ans = re.sub(r'degrees?', '', ans, flags=re.IGNORECASE)
    ans = ans.replace('\\pi', 'pi').replace('π', 'pi')
    ans = re.sub(r'\\sqrt\{([^}]+)\}', r'sqrt(\1)', ans)
    ans = re.sub(r'\\sqrt\s*([^ ]+)', r'sqrt(\1)', ans)
    ans = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'\1/\2', ans)
    ans = ans.replace('fracsqrt', 'sqrt')
    ans = ans.replace('$', '').strip(' .*')
    
    # Strip all remaining stray brackets after logic is done
    ans = ans.replace('{', '').replace('}', '').replace('**', '')
    ans = re.sub(r'[,;:\.]+$', '', ans)
    
    # Strip anomalous leftover text assignments
    ans = re.sub(r'area\\triangle\s*[A-Za-z]+', '', ans, flags=re.IGNORECASE)
    ans = ans.replace('(: m/s)', '')
    ans = ans.replace('4.0. 4.0', '4.0')
    ans = ans.replace('rsqrt', 'sqrt')

    if '=' in ans:
        ans = ans.split('=')[-1]
        
    return ans.strip()

def clean_base_model_ans(ans):
    """
    Strict extraction for Unaligned Base models (like Qwen3-4B-Base) that 
    predict the answer immediately but hallucinate continuing text/loops.
    """
    ans = ans.strip()
    
    # Strip known Base Model un-aligned hallucination loops that follow valid numbers
    ans = re.sub(r'[*]*angstroms.*', '', ans, flags=re.DOTALL)
    ans = re.sub(r'[*]*wsj.*', '', ans, flags=re.DOTALL)
    
    # Stop parsing at the first Chinese/Japanese/Korean token or newline
    ans = re.split(r'[\n\u4e00-\u9fff\u3040-\u30ff]', ans)[0]
    
    # Finally, robustly extract just the leading math sequence 
    match = re.match(r'^([-+]?[\d\./\s*+\-()pisqrt]+)', ans)
    if match:
        ans = match.group(1)

    return ans.strip()

def safe_math_eval(expr_str):
    # Pre-process implicit multiplication: "8pi" -> "8*pi", "5sqrt" -> "5*sqrt"
    expr_str = re.sub(r'(\d)(pi|sqrt)', r'\1*\2', expr_str)
    expr_str = re.sub(r'\)(pi|sqrt|\d)', r')*\1', expr_str)
    expr_str = re.sub(r'(pi)(\d)', r'\1*\2', expr_str)
    expr_str = expr_str.replace(')(', ')*(')

    allowed_names = {"pi": math.pi, "sqrt": math.sqrt, "__builtins__": {}}
    
    # Only allow safe mathematical characters
    safe_chars = set("0123456789.+-*/()pisqrt ")
    if not all(c in safe_chars for c in expr_str):
        return None
        
    import warnings
    import ast
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            # First statically parse it to ensure it's not a function call missing args
            ast.parse(expr_str)
            # Evaluate the clean string
            val = eval(expr_str, allowed_names)
            return float(val) if val is not None else None
        except Exception:
            return None

def normalize(ans):
    ans = str(ans).strip()
    
    # Pre-emptively strip units that Gemini loves to include
    ans = re.sub(r'(?:cm|mm|km|m|meters|meter|inches|inch|feet|foot|ft|nautical\smiles|nauticalmiles|nautical|miles|mile|yards|yard|yd|units|unit|square|sq|s|in)\b', '', ans, flags=re.IGNORECASE)
    ans = ans.replace('^2', '').replace('^', '').replace(' ', '')
    ans = ans.replace('(', '').replace(')', '').replace('[', '').replace(']', '')
    ans = ans.replace(':', '/')
    
    f_val = safe_math_eval(ans)
    if f_val is not None:
        if f_val == int(f_val):
            return str(int(f_val))
        return str(round(f_val, 4))
    
    return ans.lower()

def main():
    print("Loading ground truths from data/ground_truths.json...")
    with open('data/ground_truths.json', 'r') as f:
        gts = json.load(f)

    results = []
    correct = 0
    total = 0

    print("Evaluating data/geothoughts_k4_gemini3.1.jsonl...")
    with open('data/geothoughts_k4_gemini3.1.jsonl', 'r') as f:
        for line in f:
            if not line.strip(): continue
            data = json.loads(line)
            img_path = data['image_path']
            
            if img_path not in gts:
                continue
                
            gt_raw = gts[img_path]
            pred_raw = clean_gen_ans(data['reasoning'])
            
            gt_norm = normalize(gt_raw)
            pred_norm = normalize(pred_raw)
            
            # Check string norm exactly
            is_correct = (gt_norm == pred_norm)
            
            # Use math evaluation float matching as a fallback for equivalent fractions/rounding (e.g. 10/3 == 3.3333 vs 3.33)
            if not is_correct:
                gt_val = safe_math_eval(gt_raw)
                pred_val = safe_math_eval(pred_raw)
                if gt_val is not None and pred_val is not None:
                    # Allow a small math tolerance for pi/sqrt truncations
                    is_correct = math.isclose(gt_val, pred_val, rel_tol=1e-3, abs_tol=0.06)
            
            total += 1
            if is_correct:
                correct += 1
            else:
                results.append({
                    'image': img_path,
                    'gt_raw': str(gt_raw),
                    'pred_raw': pred_raw,
                    'gt_norm': gt_norm,
                    'pred_norm': pred_norm,
                    'gemini_context': data['reasoning'][-200:]
                })
                
    acc = correct / total * 100 if total > 0 else 0.0
    print(f"\n+++ RESULTS +++")
    print(f"Evaluated {total} samples.")
    print(f"Correct: {correct}")
    print(f"Accuracy: {acc:.2f}%")
    
    with open('data/eval_mismatches.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved {len(results)} mismatches to data/eval_mismatches.json")

if __name__ == "__main__":
    main()
