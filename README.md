# FindingDory-QA.

Repository for analysis of MLLM evaluation on Finding Dory benchmark.

The repository contains the following files:

1. ```evaluate_text_qa_output.py``` - Script for validating MLLM responses for each individual task type.
To use it with your own method, you need to implement a function that generates pred_lists — a list of frame lists containing the corresponding objects for navigation.
2. ```validation.json``` - GT annotation with questions and answers from FindingDory, adapted to the format for computing metrics by task categories.
3. ```answer_qa.py``` - Example script for generating answers to questions from FindingDory based on textual descriptions of video frames.
