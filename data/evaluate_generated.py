import json
import re

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
    
    if '=' in ans:
        ans = ans.split('=')[-1]
        
    return ans.strip()

def normalize(ans):
    ans = str(ans).strip()
    
    # Pre-emptively strip units that Gemini loves to include
    import re
    ans = re.sub(r'(?:cm|mm|km|m|inches|inch|feet|foot|nauticalmiles|miles|mile|units|unit|square)\b', '', ans, flags=re.IGNORECASE)
    ans = ans.replace('^2', '').replace('^', '').replace(' ', '')
    
    try:
        if '/' in ans and len(ans.split('/')) == 2:
            num, den = ans.split('/')
            f_val = float(num) / float(den)
        else:
            f_val = float(ans)
            
        if f_val == int(f_val):
            return str(int(f_val))
        return str(round(f_val, 4))
    except (ValueError, TypeError, ZeroDivisionError):
        # String normalization for pi, sqrt, etc.
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
            
            is_correct = (gt_norm == pred_norm)
            
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
