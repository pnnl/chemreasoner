"""Functions to run llama inferences."""
import logging

from pathlib import Path

from vllm import LLM, SamplingParams
from huggingface_hub import login


logging.getLogger().setLevel(logging.INFO)

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "3,4"
# use export CUDA_VISIBLE_DEVICES if running sbatch


def init_llama(
    model_dir="meta-llama/Llama-2-13b-chat-hf", num_gpus=1, **kwargs_sampling_params
):
    """Use vllm's interface to load large models over multiple GPU's"""
    login("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")
    sampling_params = SamplingParams(**kwargs_sampling_params)
    download_dir = Path("data", "model_weights") / model_dir
    download_dir.mkdir(parents=True, exist_ok=True)
    llm = LLM(
        model=model_dir,
        tensor_parallel_size=num_gpus,
        download_dir=str(download_dir),
        tokenizer="hf-internal-testing/llama-tokenizer",
    )
    return llm, sampling_params


class LlamaLLM:
    """A class to handle initializing llama."""

    default_system_prompt = ""

    def __init__(
        self,
        model_dir: Path,
        num_gpus: int,
        sampling_args: dict = {},
    ):
        """Load the model from disk and store it in this class."""
        # Apply default sampling params
        sampling_args["temperature"] = sampling_args.get("temperature", 0.8)
        sampling_args["top_p"] = sampling_args.get("top_p", 0.95)
        sampling_args["max-tokens"] = sampling_args.get("max_tokens", 1000)

        self.model_dir = model_dir
        self.num_gpus = num_gpus
        self.llm, self.sampling_params = init_llama(
            self.model_dir, self.num_gpus, **sampling_args
        )

    def __call__(
        self,
        prompts: list[str],
        system_prompts: list[str] = None,
        batch_size: int = None,
        sampling_args: dict = None,
    ):
        """Generate responses for the given prompts."""
        if sampling_args is None:
            sampling_params = self.sampling_params
        else:
            sampling_args["temperature"] = sampling_args.get("temperature", 0.8)
            sampling_args["top_p"] = sampling_args.get("top_p", 0.95)
            sampling_args["max-tokens"] = sampling_args.get("max_tokens", 1000)
            sampling_params = SamplingParams(sampling_args)

        processed_prompts = [
            self.process_prompt(p, s) for p, s in zip(prompts, system_prompts)
        ]

        return self.run_llama(processed_prompts, sampling_params, batch_size)

    def run_llama(
        self,
        processed_prompts: list[str],
        sampling_params: SamplingParams,
        batch_size: int,
    ):
        """Run the llama generation on the given processed prompts."""

        answers = []

        batch_prompts = [self.generate_prompt(x) for x in processed_prompts]
        answers = self.llm.generate(batch_prompts, sampling_params)
        for output in answers:
            generated_text = output.outputs[0].text
            answers.append(generated_text)

    def process_prompt(self, prompt: str, system_prompt: str = None):
        """Put the prompt and system prompt together."""
        sys_prompt = (
            system_prompt if system_prompt is not None else self.default_system_prompt
        )
        return (
            "<s>[INST] <<SYS>>\n"
            + sys_prompt
            + "\n<</SYS>>"
            + "\n\n"
            + prompt
            + " [/INST]"
        )


if __name__ == "__main__":
    llm = LlamaLLM(
        "meta-llama/Llama-2-13b-chat-hf",
        num_gpus=1,
    )
    llm("test1", "test2")
    # run_llama(
    #     num_gpus=1,
    #     model_dir="/qfs/projects/va_aprx/",
    #     batch_size=4,
    # )
