import torch
from gb10_load_llm import load_model_to_cuda
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "Qwen/Qwen3-32B"
MODEL_REVISION = "9216db5781bf21249d130ec9da846c4624c16137"


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run Qwen3-32B in this environment")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model, touched = load_model_to_cuda(
        AutoModelForCausalLM,
        MODEL_ID,
        revision=MODEL_REVISION,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        touch="auto",
        return_touched=True,
    )
    model.eval()

    messages = [{"role": "user", "content": "Hello, world!"}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
        )

    response_ids = output_ids[0, inputs.input_ids.shape[1] :]
    response = tokenizer.decode(response_ids, skip_special_tokens=True)
    print(f"Touched CPU pages: {touched / 1024**3:.2f} GiB")
    print(f"Qwen3-32B: {response}")


if __name__ == "__main__":
    main()
