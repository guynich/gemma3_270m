"""
Example script to run Gemma3 (270M) instruction tuned model from HuggingFace.
"""

import argparse

from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


class model:
    def __init__(self, model_name="google/gemma-3-270m-it"):
        self.model_name = model_name
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            dtype="auto",  # Use best available precision.
            attn_implementation="eager",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.pipe = pipeline(
            "text-generation", model=self.model, tokenizer=self.tokenizer
        )
        self.generation_kwargs = {
            "max_new_tokens": 256,
            "disable_compile": True
        }

    def __call__(self, text):
        """Returns model output for a given text input."""
        # Convert a text query into a prompt with the Gemma template.
        prompt = self.pipe.tokenizer.apply_chat_template(
            [{"content": f"{text}", "role": "user"}],
            tokenize=False,
            add_generation_prompt=True,
        )
        outputs = self.pipe(prompt, **self.generation_kwargs)
        return outputs[0]["generated_text"][len(prompt) :].strip()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_token", required=True)
    parser.add_argument("--prompt", default="What causes climate change?", type=str)
    return parser.parse_args()


def main():
    args = parse_args()

    login(token=args.hf_token)

    model_runner = model()

    print(f"Model:        {model_runner.model_name}")
    print(f"Precision:    {model_runner.model.dtype}")
    print("=" * 80)
    print(f"Input prompt: {args.prompt}")
    print(model_runner(args.prompt))


if __name__ == "__main__":
    main()
