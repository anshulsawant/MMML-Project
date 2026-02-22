
import json
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def clean_prediction(prediction):
    if not isinstance(prediction, str):
        return None
    match = re.match(r"^\s*([A-D])", prediction)
    if match:
        return match.group(1)
    return None

def load_predictions(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def load_ground_truth(test_dir):
    ground_truth = {}
    for subdir in os.listdir(test_dir):
        data_path = os.path.join(test_dir, subdir, 'data.json')
        if os.path.exists(data_path):
            with open(data_path, 'r') as f:
                data = json.load(f)
                ground_truth[subdir] = {
                    'answer': data.get('answer'),
                    'problem_type_graph': data.get('problem_type_graph', []),
                    'problem_type_goal': data.get('problem_type_goal', [])
                }
    return ground_truth

def analyze_accuracies(predictions, ground_truth):
    results = []
    for pred in predictions:
        problem_id = pred['problem_id']
        if problem_id in ground_truth:
            gt = ground_truth[problem_id]
            cleaned_pred = clean_prediction(pred['prediction'])
            is_correct = cleaned_pred == gt['answer']
            results.append({
                'problem_id': problem_id,
                'prediction': cleaned_pred,
                'ground_truth': gt['answer'],
                'is_correct': is_correct,
                'problem_type_graph': gt['problem_type_graph'],
                'problem_type_goal': gt['problem_type_goal']
            })
    return results

def run_analysis(group_name, file_map, ground_truth):
    print(f"--- Analyzing Group: {group_name} ---")
    dfs = []
    for label, filename in file_map.items():
        if os.path.exists(filename):
            preds = load_predictions(filename)
            results = analyze_accuracies(preds, ground_truth)
            df = pd.DataFrame(results)
            df['variant'] = label
            dfs.append(df)
            
            accuracy = df['is_correct'].mean()
            print(f"{label} Accuracy: {accuracy:.2%}")
        else:
            print(f"Warning: File {filename} not found.")
            
    if not dfs:
        print("No data found for this group.")
        return

    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Overall Accuracy Comparison
    plt.figure(figsize=(8, 6))
    overall_acc = combined_df.groupby('variant')['is_correct'].mean().reset_index()
    sns.barplot(data=overall_acc, x='variant', y='is_correct')
    plt.title(f'Overall Prediction Accuracy Comparison - {group_name}')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    output_path = f'molmo2/accuracy_comparison_{group_name}.png'
    plt.savefig(output_path)
    print(f"Saved overall accuracy comparison to {output_path}")
    plt.close()

    # Accuracy by graph type
    df_graph = combined_df.explode('problem_type_graph')
    graph_accuracy = df_graph.groupby(['problem_type_graph', 'variant'])['is_correct'].mean().reset_index()
    
    plt.figure(figsize=(12, 8))
    sns.barplot(data=graph_accuracy, x='problem_type_graph', y='is_correct', hue='variant')
    plt.title(f'Accuracy by Problem Graph Type - {group_name}')
    plt.ylabel('Accuracy')
    plt.xlabel('Problem Graph Type')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1)
    plt.tight_layout()
    output_path_graph = f'molmo2/accuracy_by_graph_type_{group_name}.png'
    plt.savefig(output_path_graph)
    print(f"Saved accuracy by graph type to {output_path_graph}")
    plt.close()

    # Accuracy by goal type
    df_goal = combined_df.explode('problem_type_goal')
    goal_accuracy = df_goal.groupby(['problem_type_goal', 'variant'])['is_correct'].mean().reset_index()

    plt.figure(figsize=(12, 8))
    sns.barplot(data=goal_accuracy, x='problem_type_goal', y='is_correct', hue='variant')
    plt.title(f'Accuracy by Problem Goal Type - {group_name}')
    plt.ylabel('Accuracy')
    plt.xlabel('Problem Goal Type')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1)
    plt.tight_layout()
    output_path_goal = f'molmo2/accuracy_by_goal_type_{group_name}.png'
    plt.savefig(output_path_goal)
    print(f"Saved accuracy by goal type to {output_path_goal}")
    plt.close()
    print("\n")

def main():
    ground_truth = load_ground_truth('test')

    groups = [
        ('direct_vs_thinking', {
            'Direct': 'molmo2/predictions_direct.json',
            'Thinking': 'molmo2/predictions_thinking.json'
        }),
        ('text_vs_graph', {
            'Text-Only': 'molmo2/predictions_text_only.json',
            'JSON Graph': 'molmo2/predictions_json_graph.json'
        }),
        ('all_variants', {
            'Direct': 'molmo2/predictions_direct.json',
            'Thinking': 'molmo2/predictions_thinking.json',
            'Text-Only': 'molmo2/predictions_text_only.json',
            'JSON Graph': 'molmo2/predictions_json_graph.json'
        })
    ]

    for group_name, file_map in groups:
        run_analysis(group_name, file_map, ground_truth)


if __name__ == '__main__':
    main()
