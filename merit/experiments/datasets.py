"""Dataset loading and instruction templates for MERIT experiments."""
import random
from typing import List, Dict


# Instruction templates for different tasks
INSTRUCTION_TEMPLATES = {
    "multiple_choice": """Answer the following multiple choice question. Choose the best answer from the given options and explain your reasoning briefly.

Question: {question}

Instructions: Select ONE answer from the options above. Start your response with the letter of your choice (A, B, C, or D), then briefly explain why.""",

    "reasoning": """Please solve the following problem step by step:

{question}

Show your reasoning process clearly.""",

    "math_reasoning": """Solve this math problem step by step:

{question}

Show all your work. Give your final numerical answer after ####.""",

    "default": """Please answer the following question:

{question}

Provide a clear and concise answer."""
}


def load_dataset(benchmark: str, sample_size: int, seed: int,
                 use_instruction_format: bool = True) -> List[Dict]:
    """Load and sample dataset.

    Parameters
    ----------
    benchmark : str
        Name of the benchmark (arc, hellaswag, truthfulqa, mmlu_logic,
        gsm8k, bbh).
    sample_size : int
        Number of samples to draw.  If <= 0, use the full dataset.
    seed : int
        Random seed for reproducible sampling.
    use_instruction_format : bool
        Whether to wrap prompts with instruction templates.

    Returns
    -------
    list[dict]
        Formatted dataset items with keys: prompt, reference,
        reference_letter, task_type.
    """
    try:
        from datasets import load_dataset as hf_load_dataset

        # Load dataset based on benchmark name
        if benchmark == "arc":
            ds = hf_load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
        elif benchmark == "hellaswag":
            ds = hf_load_dataset("Rowan/hellaswag", split="validation")
        elif benchmark == "truthfulqa":
            ds = hf_load_dataset("truthfulqa/truthful_qa", "multiple_choice", split="validation")
        elif benchmark == "mmlu_logic":
            ds = hf_load_dataset("cais/mmlu", "formal_logic", split="test")
        elif benchmark == "gsm8k":
            ds = hf_load_dataset("openai/gsm8k", "main", split="test")
        elif benchmark == "bbh":
            ds = hf_load_dataset("lukaemon/bbh", "logical_deduction_five_objects", split="test")
        else:
            print(f"      Unknown benchmark: {benchmark}")
            return []

        # Sample the dataset
        random.seed(seed)

        total_size = len(ds)

        # If sample_size <= 0, use full dataset
        if sample_size <= 0:
            sample_size = total_size
        else:
            sample_size = min(sample_size, total_size)

        indices = random.sample(range(total_size), sample_size)
        sampled_data = [ds[i] for i in indices]

        # Convert to standard format with instruction templates
        formatted_data = []
        for item in sampled_data:
            if benchmark == "arc":
                choices_text = "\n".join([f"({chr(65+i)}) {choice}" for i, choice in enumerate(item["choices"]["text"])])
                answer_key = item["answerKey"]
                answer_idx = item["choices"]["label"].index(answer_key)
                reference_text = item["choices"]["text"][answer_idx]

                # Raw question with choices
                raw_question = f"{item['question']}\n\nOptions:\n{choices_text}"

                # Apply instruction template if enabled
                if use_instruction_format:
                    prompt = INSTRUCTION_TEMPLATES["multiple_choice"].format(question=raw_question)
                else:
                    prompt = raw_question

                formatted_data.append({
                    "prompt": prompt,
                    "reference": reference_text,
                    "reference_letter": answer_key,
                    "task_type": "multiple_choice"
                })

            elif benchmark == "hellaswag":
                context = f"{item['ctx']}\n{item['activity_label']}"
                endings = "\n".join([f"({chr(65+i)}) {ending}" for i, ending in enumerate(item["endings"])])
                raw_question = f"{context}\n\nChoose the best continuation:\n{endings}"

                if use_instruction_format:
                    prompt = INSTRUCTION_TEMPLATES["multiple_choice"].format(question=raw_question)
                else:
                    prompt = raw_question

                formatted_data.append({
                    "prompt": prompt,
                    "reference": item["endings"][int(item["label"])],
                    "reference_letter": chr(65 + int(item["label"])),
                    "task_type": "multiple_choice"
                })

            elif benchmark == "truthfulqa":
                choices = item["mc1_targets"]["choices"]
                labels = item["mc1_targets"]["labels"]
                correct_idx = labels.index(1)  # Find the correct answer index

                choices_text = "\n".join([f"({chr(65+i)}) {choice}" for i, choice in enumerate(choices)])
                raw_question = f"{item['question']}\n\nOptions:\n{choices_text}"

                if use_instruction_format:
                    prompt = INSTRUCTION_TEMPLATES["multiple_choice"].format(question=raw_question)
                else:
                    prompt = raw_question

                formatted_data.append({
                    "prompt": prompt,
                    "reference": choices[correct_idx],
                    "reference_letter": chr(65 + correct_idx),
                    "task_type": "multiple_choice"
                })

            elif benchmark == "mmlu_logic":
                choices_text = "\n".join([f"({chr(65+i)}) {choice}" for i, choice in enumerate(item["choices"])])
                raw_question = f"{item['question']}\n\nOptions:\n{choices_text}"

                if use_instruction_format:
                    prompt = INSTRUCTION_TEMPLATES["multiple_choice"].format(question=raw_question)
                else:
                    prompt = raw_question

                formatted_data.append({
                    "prompt": prompt,
                    "reference": item["choices"][item["answer"]],
                    "reference_letter": chr(65 + item["answer"]),
                    "task_type": "multiple_choice"
                })

            elif benchmark == "gsm8k":
                raw_question = item["question"]

                if use_instruction_format:
                    prompt = INSTRUCTION_TEMPLATES["math_reasoning"].format(question=raw_question)
                else:
                    prompt = raw_question

                # Extract the final numerical answer after ####
                answer_text = item["answer"]
                if "####" in answer_text:
                    reference = answer_text.split("####")[-1].strip()
                else:
                    reference = answer_text

                formatted_data.append({
                    "prompt": prompt,
                    "reference": reference,
                    "reference_letter": None,
                    "task_type": "math_reasoning"
                })

            elif benchmark == "bbh":
                raw_question = item["input"]

                if use_instruction_format:
                    prompt = INSTRUCTION_TEMPLATES["reasoning"].format(question=raw_question)
                else:
                    prompt = raw_question

                formatted_data.append({
                    "prompt": prompt,
                    "reference": item["target"],
                    "reference_letter": None,
                    "task_type": "reasoning"
                })

        return formatted_data

    except Exception as e:
        print(f"      Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return []
