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
        
        pred_ans = extract_answer(generated_string)
        
        if i == 0:  # Print the first generation sequence in each batch dynamically to avoid spam
            print(f"\n[GRPO DIAGNOSTIC LOG]")
            print(f"Raw Output    : {repr(generated_string)}")
            print(f"Extracted Ans : {repr(pred_ans)}")
            print(f"Ground Truth  : {repr(gt_ans)}")
            print(f"-----------------------\n")
            
        if pred_ans and gt_ans:
            # We enforce raw string evaluations explicitly ignoring whitespace and capitalization
            if pred_ans.strip().lower() == gt_ans.strip().lower():
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        else:
            rewards.append(0.0)
            
    return rewards


