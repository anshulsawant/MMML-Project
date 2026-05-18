import re
import json

def extract_answer(text) -> str:
    """
    Extracts the raw mathematical string directly natively evaluating without box encasings.
    """
    if not isinstance(text, str):
        if isinstance(text, list) and len(text) > 0 and isinstance(text[-1], dict):
            text = text[-1].get("content", "")
        else:
            text = str(text)
            
    return text.replace("<|im_end|>", "").strip()
def accuracy_reward_func(prompts, completions, target_math=None, **kwargs) -> list[float]:
    """
    Computes +1.0 binary reward dynamically evaluating parsed sequence logic against the raw target math bounds gracefully mapping kwargs.
    """
    rewards = []
    
    for i in range(len(completions)):
        generated_string = completions[i]
        gt_ans = str(target_math[i]) if target_math is not None and i < len(target_math) else ""
        
        from evaluate_generated import clean_base_model_ans, safe_math_eval, normalize
        
        pred_ans = extract_answer(generated_string)
        
        if i == 0:  # Print the first generation sequence in each batch dynamically to avoid spam
            print(f"\n[GRPO DIAGNOSTIC LOG]")
            print(f"Raw Output    : {repr(generated_string)}")
            print(f"Extracted Ans : {repr(pred_ans)}")
            print(f"Ground Truth  : {repr(gt_ans)}")
            print(f"-----------------------\n")
            
        if pred_ans and gt_ans:
            # We enforce numeric bounds explicitly ignoring generative paragraph text securely!
            pred_raw = clean_base_model_ans(pred_ans)
            gt_norm = normalize(gt_ans)
            pred_norm = normalize(pred_raw)
            
            is_correct = (gt_norm == pred_norm)
            
            if not is_correct:
                import math
                gt_val = safe_math_eval(gt_ans)
                pred_val = safe_math_eval(pred_raw)
                if gt_val is not None and pred_val is not None:
                    is_correct = math.isclose(gt_val, pred_val, rel_tol=1e-3, abs_tol=0.06)
                    
            if is_correct:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        else:
            rewards.append(0.0)
            
    return rewards


