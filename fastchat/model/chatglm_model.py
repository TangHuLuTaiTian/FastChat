import torch


@torch.inference_mode()
def chatglm_generate_stream(
    model, tokenizer, params, device, context_len=2048, stream_interval=2
):
    """Generate text using model's chat api"""
    prompt = params["prompt"]
    max_new_tokens = int(params.get("max_new_tokens", 256))
    temperature = float(params.get("temperature", 1.0))
    top_p = float(params.get("top_p", 1.0))
    repetition_penalty = float(params.get("repetition_penalty", 1.0))
    echo = params.get("echo", True)

    gen_kwargs = {
        # "max_new_tokens": max_new_tokens,  disabled due to a warning.
        "do_sample": True if temperature > 1e-5 else False,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
        "logits_processor": None,
    }
    if temperature > 1e-5:
        gen_kwargs["temperature"] = temperature

    input_ids = tokenizer([prompt], return_tensors="pt").input_ids.to(model.device)
    max_src_len = context_len - max_new_tokens - 8
    input_ids = input_ids[-max_src_len:]
    input_echo_len = len(input_ids)

    output = ""
    i = 0
    for i, (response, new_hist) in enumerate(
        model.stream_generate(input_ids, **gen_kwargs)
    ):
        if echo:
            output = prompt + " " + response
        else:
            output = response

        yield {
            "text": output,
            "usage": {
                "prompt_tokens": input_echo_len,
                "completion_tokens": i,
                "total_tokens": input_echo_len + i,
            },
            "finish_reason": None,
        }

    # TODO: ChatGLM stop when it reach max length
    # Only last stream result contains finish_reason, we set finish_reason as stop
    ret = {
        "text": output,
        "usage": {
            "prompt_tokens": input_echo_len,
            "completion_tokens": i,
            "total_tokens": input_echo_len + i,
        },
        "finish_reason": "stop",
    }
    yield ret
