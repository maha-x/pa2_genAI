PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n"


def format_prompt(instruction: str) -> str:
    """Wrap a raw instruction in the template used for ALL stages."""
    return PROMPT_TEMPLATE.format(instruction=instruction.strip())
