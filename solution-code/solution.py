"""
Day 1 — LLM API Foundation
AICB-P1: AI Practical Competency Program, Phase 1

Instructions:
    1. Fill in every section marked with TODO.
    2. Do NOT change function signatures.
    3. Copy this file to solution/solution.py when done.
    4. Run: pytest tests/ -v
"""

import os
import time
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Estimated costs per 1M INPUT & OUTPUT tokens (USD) as of March 2026
# Vietnamese text generally consumes ~1.5x - 2.0x more tokens than English due to Unicode/diacritics.
# ---------------------------------------------------------------------------
PRICING_1M_TOKENS = {
    "gpt-4o": {"input": 5.00, "output": 20.00},
    "gpt-4o-mini": {"input": 0.150, "output": 0.600},
    "gemini-2.5-flash": {"input": 0.075, "output": 0.300},
    "gemini-2.5-pro": {"input": 1.25, "output": 5.00},
    "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku": {"input": 0.80, "output": 4.00},
}

# Standard Model Identifiers
OPENAI_MODEL = "gpt-4o"
OPENAI_MINI_MODEL = "gpt-4o-mini"
GEMINI_MODEL = "gemini-2.5-flash"
ANTHROPIC_MODEL = "claude-3-5-haiku"


# ---------------------------------------------------------------------------
# Task 1 — Call OpenAI (GPT-4o)
# ---------------------------------------------------------------------------
def call_openai(
    prompt: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """
    Call the OpenAI Chat Completions API and return the response text, latency,
    and token usage stats.

    Args:
        prompt:      The user message to send.
        model:       The OpenAI model to use (default: gpt-4o).
        temperature: Sampling temperature (0.0 – 2.0).
        top_p:       Nucleus sampling threshold.
        max_tokens:  Maximum number of tokens to generate.

    Returns:
        A tuple of:
            - response_text (str)
            - latency_seconds (float)
            - usage (dict with keys: 'input_tokens', 'output_tokens')

    Hint:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        # response.usage contains input_tokens and output_tokens (prompt_tokens/completion_tokens)
    """
    # TODO: Import OpenAI, instantiate client, call chat.completions.create with parameters,
    #       measure execution start/end time, extract text and token usage, and return them.
    from openai import OpenAI
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)
    start = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    latency = time.time() - start
    usage = response.usage
    input_tokens = usage.prompt_tokens if usage is not None else 0
    output_tokens = usage.completion_tokens if usage is not None else 0

    return response.choices[0].message.content or "", latency, {"input_tokens": input_tokens, "output_tokens": output_tokens}
    raise NotImplementedError("Implement call_openai")


# ---------------------------------------------------------------------------
# Task 2 — Call Google Gemini 2.5 (Standard Practical Model)
# ---------------------------------------------------------------------------
def call_gemini(
    prompt: str,
    model: str = GEMINI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """
    Call the Google Gemini API (using Gemini 2.5 Flash as standard) and return
    the response text, latency, and token usage stats.

    Args:
        prompt:      The user message to send.
        model:       The Gemini model to use (default: gemini-2.5-flash).
        temperature: Sampling temperature.
        top_p:       Nucleus sampling threshold.
        max_tokens:  Maximum number of tokens to generate.

    Returns:
        A tuple of:
            - response_text (str)
            - latency_seconds (float)
            - usage (dict with keys: 'input_tokens', 'output_tokens')

    Hint:
        Option A (New Google GenAI SDK):
            from google import genai
            from google.genai import types
            client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
            # Configure using types.GenerateContentConfig
            
        Option B (Legacy Google GenerativeAI SDK):
            import google.generativeai as genai
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            model_inst = genai.GenerativeModel(model)
            # Configure using genai.types.GenerationConfig
            
        Ensure your usage dictionary extracts 'input_tokens' and 'output_tokens' 
        from the response metadata (e.g. response.usage_metadata).
    """
    # TODO: Initialize Gemini client, set config parameters, call generate_content,
    #       measure latency, extract response text and usage metadata, and return the tuple.
    from google import genai
    from google.genai import types
    start = time.time()
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    config = types.GenerateContentConfig(
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_tokens
    )

    if "flash" in model:
        config.thinking_config = types.ThinkingConfig(thinking_budget=0)
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config
    )
    latency = time.time() - start
    usage = response.usage_metadata

    input_tokens = (usage.prompt_token_count or 0) if usage is not None else 0
    output_tokens = (usage.candidates_token_count or 0) if usage is not None else 0
    response_text = response.text or ""
    return response_text, latency, {"input_tokens": input_tokens, "output_tokens": output_tokens}
    raise NotImplementedError("Implement call_gemini")


# ---------------------------------------------------------------------------
# Task 3 — Call Anthropic Claude (Exploratory track)
# ---------------------------------------------------------------------------
def call_anthropic(
    prompt: str,
    model: str = ANTHROPIC_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """
    Call the Anthropic Claude API (using Claude 3.5 Haiku as default) and return
    the response text, latency, and token usage stats.

    Args:
        prompt:      The user message to send.
        model:       The Claude model to use (default: claude-3-5-haiku).
        temperature: Sampling temperature (0.0 - 1.0).
        top_p:       Nucleus sampling threshold.
        max_tokens:  Maximum output tokens.

    Returns:
        A tuple of:
            - response_text (str)
            - latency_seconds (float)
            - usage (dict with keys: 'input_tokens', 'output_tokens')

    Hint:
        import anthropic
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        # response.usage contains input_tokens and output_tokens
    """
    # TODO: Initialize Anthropic client, create message, measure latency,
    #       extract content text and usage statistics, and return the tuple.
    import anthropic
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    start = time.time()
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
        extra_body={"temperature": temperature, "top_p": top_p},
    )
    latency = time.time() - start
    text_parts: list[str] = []
    for block in response.content:
        block_text = getattr(block, "text", None)
        if isinstance(block_text, str):
            text_parts.append(block_text)
    response_text = "".join(text_parts)
    usage = response.usage
    input_tokens = (usage.input_tokens or 0) if usage is not None else 0
    output_tokens = (usage.output_tokens or 0) if usage is not None else 0

    return response_text, latency, {"input_tokens": input_tokens, "output_tokens": output_tokens}
    raise NotImplementedError("Implement call_anthropic")


# ---------------------------------------------------------------------------
# Task 4 — Compare Models (OpenAI GPT-4o vs OpenAI Mini vs Gemini 2.5 Flash)
# ---------------------------------------------------------------------------
def compare_models(prompt: str) -> dict:
    """
    Call OpenAI (gpt-4o), OpenAI Mini (gpt-4o-mini), and Gemini 2.5 Flash (gemini-2.5-flash)
    with the same prompt and return a structured comparison dictionary.

    Calculate the exact USD token cost for input + output using the prices in PRICING_1M_TOKENS.

    Args:
        prompt: The user message to send to all models.

    Returns:
        A dictionary containing:
            - "gpt4o": { "response": str, "latency": float, "cost": float, "input_tokens": int, "output_tokens": int }
            - "gpt4o_mini": { "response": str, "latency": float, "cost": float, "input_tokens": int, "output_tokens": int }
            - "gemini_flash": { "response": str, "latency": float, "cost": float, "input_tokens": int, "output_tokens": int }
    """
    # TODO: Call call_openai with default gpt-4o model
    # TODO: Call call_openai with gpt-4o-mini model
    # TODO: Call call_gemini with default gemini-2.5-flash model
    # TODO: Calculate costs exactly based on input and output token counts using PRICING_1M_TOKENS
    #       Formula: Cost = (input_tokens * input_rate_per_1M + output_tokens * output_rate_per_1M) / 1,000,000
    # TODO: Assemble and return the comparison dictionary.
    def _build_entry(result: tuple[str, float, dict], model: str) -> dict:
        text, latency, usage = result
        rates = PRICING_1M_TOKENS[model]
        cost = (
            usage["input_tokens"] * rates["input"]
            + usage["output_tokens"] * rates["output"]
        ) / 1_000_000
        return {
            "response": text,
            "latency": latency,
            "cost": cost,
            "input_tokens": usage["input_tokens"],
            "output_tokens": usage["output_tokens"],
        }
    gpt4o_result = call_openai(prompt)
    gpt4o_mini_result = call_openai(prompt, model=OPENAI_MINI_MODEL)
    gemini_result = call_gemini(prompt)

    return {
        "gpt4o": _build_entry(gpt4o_result, OPENAI_MODEL),
        "gpt4o_mini": _build_entry(gpt4o_mini_result, OPENAI_MINI_MODEL),
        "gemini_flash": _build_entry(gemini_result, GEMINI_MODEL),
    }
    raise NotImplementedError("Implement compare_models")


# ---------------------------------------------------------------------------
# Task 5 — Streaming chatbot with Gemini 2.5 (Focus Model)
# ---------------------------------------------------------------------------
def streaming_chatbot() -> None:
    """
    Run an interactive streaming chatbot in the terminal using Gemini 2.5.

    Behaviour:
        - Streams response tokens from Gemini 2.5 Flash as they arrive.
        - Maintains the last 3 turns of conversation history for context.
        - Typing 'quit' or 'exit' ends the session.

    Hints:
        - Maintain a history list of conversation turns.
        - Check how to stream responses using client.chats or model.generate_content(..., stream=True).
        - Keep history limited to the last 3 turns to optimize context window and costs.
    """
    # TODO: Setup interactive session, prompt user for input, stream response, and update history.
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    max_history_messages = 6  # last 3 turns = 3 user + 3 model messages
    history: list[types.Content] = []

    def _chat_stream():
        """
        Run the chat loop and yield every text chunk Gemini streams back.
        The `for chunk in response_stream` loop below prints the chunks, so the
        reading of input, the streaming and the history all happen here.
        """
        nonlocal history
        while True:
            try:
                user_input = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if user_input.lower() in ("quit", "exit"):
                break
            if not user_input:
                continue

            contents = history + [
                types.Content(role="user", parts=[types.Part(text=user_input)])
            ]
            print("Gemini: ", end="", flush=True)
            reply_parts: list[str] = []
            try:
                stream = client.models.generate_content_stream(
                    model=GEMINI_MODEL,
                    contents=contents,
                )
                for chunk in stream:
                    if chunk.text:  # some chunks (e.g. the last one) carry no text
                        reply_parts.append(chunk.text)
                        yield chunk
            except Exception as e:
                # A failed turn is reported and left out of the history.
                print(f"\n[Error] {e}")
                continue
            print()

            reply = "".join(reply_parts)
            if not reply:
                print("[No text in the response]")
                continue
            contents.append(types.Content(role="model", parts=[types.Part(text=reply)]))
            history = contents[-max_history_messages:]
        print("Goodbye!")

    response_stream = _chat_stream()
    for chunk in response_stream:
        print(chunk.text, end="", flush=True)
    return
    raise NotImplementedError("Implement streaming_chatbot")


# ---------------------------------------------------------------------------
# Bonus Task A — Retry with exponential backoff
# ---------------------------------------------------------------------------
def retry_with_backoff(
    fn: Callable[[], Any],
    max_retries: int = 3,
    base_delay: float = 0.1,
) -> Any:
    """
    Call fn(). If it raises an exception, retry up to max_retries times
    with exponential backoff (delay = base_delay * 2^attempt).

    Args:
        fn:          Zero-argument callable to execute.
        max_retries: Maximum number of retry attempts.
        base_delay:  Initial delay in seconds before the first retry.

    Returns:
        The return value of fn() on success.

    Raises:
        The last exception raised by fn() after all retries are exhausted.
    """
    # TODO: implement retry loop with exponential backoff
    # 1 first try + max_retries retries; the delay doubles each time:
    # base_delay, base_delay*2, base_delay*4, ...
    max_retries = max(0, max_retries)
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception:
            if attempt == max_retries:
                raise  # out of retries: re-raise the last exception
            time.sleep(base_delay * (2 ** attempt))
    raise NotImplementedError("Implement retry_with_backoff")


# ---------------------------------------------------------------------------
# Bonus Task B — Batch compare
# ---------------------------------------------------------------------------
def batch_compare(prompts: list[str]) -> list[dict]:
    """
    Run compare_models on each prompt in the list.

    Args:
        prompts: List of prompt strings.

    Returns:
        List of dicts, each being the compare_models result with an extra
        key "prompt" containing the original prompt string.
    """
    # TODO: iterate over prompts, call compare_models, and inject the original "prompt".
    results = []
    for prompt in prompts:
        result = compare_models(prompt)
        result["prompt"] = prompt
        results.append(result)
    return results
    raise NotImplementedError("Implement batch_compare")


# ---------------------------------------------------------------------------
# Bonus Task C — Format comparison table
# ---------------------------------------------------------------------------
def format_comparison_table(results: list[dict]) -> str:
    """
    Format a list of batch compare results as a readable Markdown table string.

    Args:
        results: List of dicts as returned by batch_compare.

    Returns:
        A beautiful Markdown table string with columns:
        | Prompt | Model | Response (truncated) | Latency | Tokens (In/Out) | Cost (USD) |
    """
    # TODO: Build and return the formatted table string. Truncate response to 50 chars for clean display.
    display_names = {
        "gpt4o": "GPT-4o",
        "gpt4o_mini": "GPT-4o-Mini",
        "gemini_flash": "Gemini-Flash",
    }
    max_response_chars = 50

    def _cell(value: Any) -> str:
        # Keep each value on one line and escape "|" so it can't break the table.
        return " ".join(str(value).split()).replace("|", "\\|")

    lines = [
        "| Prompt | Model | Response (truncated) | Latency | Tokens (In/Out) | Cost (USD) |",
        "|---|---|---|---|---|---|",
    ]
    for result in results:
        prompt = result.get("prompt", "")
        for key, stats in result.items():
            if key == "prompt" or not isinstance(stats, dict):
                continue
            response = _cell(stats.get("response", ""))
            if len(response) > max_response_chars:
                response = response[:max_response_chars] + "..."
            lines.append(
                f"| {_cell(prompt)} | {display_names.get(key, key)} | {response} "
                f"| {stats.get('latency', 0.0):.2f}s "
                f"| {stats.get('input_tokens', 0)}/{stats.get('output_tokens', 0)} "
                f"| ${stats.get('cost', 0.0):.6f} |"
            )
    return "\n".join(lines)
    raise NotImplementedError("Implement format_comparison_table")


# ---------------------------------------------------------------------------
# Entry point for manual testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Load API keys from the project's .env file (python-dotenv is in requirements.txt).
    # Variables already set in the environment are not overwritten.
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    print("=== Model Comparison Test ===")
    test_prompt = "Hãy giải thích sự khác biệt giữa temperature và top_p bằng tiếng Việt ngắn gọn trong 2 câu."
    try:
        # Note: Requires valid API keys set in environment variables
        result = compare_models(test_prompt)
        for model_name, stats in result.items():
            print(f"\n[{model_name.upper()}]")
            print(f"Latency: {stats['latency']:.2f}s | Cost: ${stats['cost']:.6f}")
            print(f"Tokens: {stats['input_tokens']} in / {stats['output_tokens']} out")
            print(f"Response: {stats['response']}")
    except Exception as e:
        print(f"Skipping live API comparison test: {e}")
        print("Set your API keys to run manual tests.")

    print("\n=== Starting Gemini 2.5 Chatbot (type 'quit' to exit) ===")
    try:
        streaming_chatbot()
    except Exception as e:
        print(f"Chatbot failed to start: {e}")
