import json
import pandas as pd
import os
from typing import Dict, List
import ast

import numpy as np

def calculate_relaxed_match(pred_lists: List[List[int]], gt_lists: List[List[int]]) -> float:
    """
    Calculate relaxed matching score as a product of precision scores for each sublist.

    For each predicted sublist and corresponding ground truth sublist:
    - Calculates precision as (number of predicted elements in ground truth) / (number of predicted elements)
    - Returns product of precision scores across all sublists

    Returns:
    - 0.0 if number of sublists don't match
    - Product of precision scores (between 0.0 and 1.0) otherwise
    """
    # Check if number of sublists match
    if len(pred_lists) != len(gt_lists):
        return 0.0

    # Check each corresponding sublist pair
    precision_all_goals = []
    for pred_sublist, gt_sublist in zip(pred_lists, gt_lists):
        # If none of the predicted elements appear in ground truth sublist, return 0
        if len(pred_sublist) == 0 and len(gt_sublist) == 0:
            precision = 1.0
        elif len(pred_sublist) == 0 or len(gt_sublist) == 0:
            precision = 0.0
        else:
            precision = sum(pred_elem in gt_sublist for pred_elem in pred_sublist) / len(pred_sublist)
            precision_all_goals.append(precision)

    # multiply precision of all goals
    return float(np.prod(precision_all_goals))


def evaluate_results(results_dir: str, gt_file: str) -> Dict:
    """Evaluate both exact and relaxed matching metrics from results file."""
    
    with open(gt_file, "r") as f:
        anns = json.load(f)

    total_examples = len(anns)
    exact_matches = 0
    relaxed_scores = []  # Changed from counter to list of scores

    finished = 0

    ann_stats = {
        "Single-Goal Spatial Tasks": {
            "Room Visitation": [], 
            "Interaction": [], 
            "Object Recall": [], 
            "Conditional Interaction": [], 
            "Spatial Relationship": [], 
            "Object Attributes": [],
        },
        "Multi-Goal Tasks": {
            "Unordered Revisitation": [], 
            "Ordered Revisitation": [], 
        },
        "Single-Goal Temporal Tasks": {
            "Interaction Order": [], 
            "Duration Tracking": [],
            "Time-Based": []
        }
    }
    for i, ann in enumerate(anns):
        try:
            # Get ground truth and model output
            if ann["ep_id"] not in ["ep_2", "ep_5", "ep_6", "ep_9"]:
                continue
            gt_list = ann["answer"]

            ####### PARSING LIST OF PREDICTED FRAMES, pred_list is List[List]
            # try:
            #     with open(os.path.join(results_dir, f"{ann['id']}.json"), "r") as f:
            #         pred_list = json.load(f)["frame_indices"]
            # except:
            #     pred_list = None
            ######

            # Calculate exact match
            if gt_list == pred_list:
                exact_matches += 1

            # Calculate relaxed match
            gt_lists = gt_list
            pred_lists = pred_list

            # If either parsing returned empty list (parsing failure), treat as 0 for relaxed metric
            if not gt_lists or not pred_lists:
                # relaxed_scores.append(0.0)
                # ann_stats[ann["high_level_category"]][ann["low_level_category"]].append(0.0)
                # finished += 1
                continue

            # Store the actual precision score
            relaxed_score = calculate_relaxed_match(pred_lists, gt_lists)
            relaxed_scores.append(relaxed_score)
            ann_stats[ann["high_level_category"]][ann["low_level_category"]].append(relaxed_score)

            finished += 1

        except Exception as e:
            # Treat exceptions as 0 for relaxed metric
            # relaxed_scores.append(0.0)
            print(f"Error processing example: {e}")
            continue

    print("Finished:", finished)
    # Calculate accuracies
    exact_accuracy = exact_matches / total_examples if total_examples > 0 else 0
    avg_relaxed_score = np.mean(relaxed_scores) if relaxed_scores else 0

    results_dict = {
        "total_examples": total_examples,
        "exact_matches": exact_matches,
        "avg_relaxed_score": avg_relaxed_score,
        "exact_accuracy": exact_accuracy,
    }

    for high_level_k in ann_stats:
        results_dict[high_level_k] = {}
        for low_level_k in ann_stats[high_level_k]:
            results_dict[high_level_k][low_level_k] = np.mean(ann_stats[high_level_k][low_level_k])

    return results_dict


def main():
    """
    Main function to aggregate VLM checkpoint performance.
    """

    results_dir = "text_qa/answers_time_vl_image_16/"
    gt_file = "validation.json"

    results = evaluate_results(results_dir, gt_file)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
