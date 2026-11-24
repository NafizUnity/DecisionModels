"""Run the standalone ICD-10-CA picker on an ordered candidate list.

Prompt and prediction logic come from workflow/scripts/rag/code_entities_rag.py.
The Transformers and Unsloth wrappers and model-family parsers come from that
directory's llm/ modules. CIHI matching comes from coding_standards.py. All of
that code is included below, so this script can run outside the repository.

RealData copy of GeminiOps/run_rag_picker.py. Changes from the uploaded file (search "RealData:"):
  - run.use_note: false gives a prompt without the clinical-note section (the default keeps the original prompt).
  - model.stop_tokens: "generation_config" (default) stops at the model's own end tokens (Gemma 4: 1, 106, 50);
    "tokenizer_eos" keeps the original behaviour (tokenizer eos only: Gemma runs on to max_new_tokens).
  - an unparseable reply gives an empty pick (status no_answer) instead of the first code found in the prompt;
    a pick outside the candidate list is kept and marked not_in_list.
  - each prediction row also holds entity_id, status and the raw reply.
  - picker.format: "requests" reads a JSONL of entities with ready candidate lists (picker_build.py --format requests);
    "native" (default) reads the consolidated JSON as before. A candidate list may be shorter than candidate_pool.
  - several --config files in one call share one model load; a config whose output exists is skipped.
  - model.batch_size > 1 (transformers backend) answers several entities per generate call; 1 (default) is unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import yaml
from tqdm import tqdm

if TYPE_CHECKING:
    import torch

log = logging.getLogger(__name__)
RANKING_KEY = "external_ranking"


# Picker prompts and prediction helpers

SYSTEM_PROMPT = """\
You are an expert medical coder. Your task is to assign the single most \
appropriate ICD-10-CA code to a clinical entity from the provided candidate list.

Rules:
1. You MUST pick exactly one code from the candidate list — do not invent codes.
2. Base your decision ONLY on what the entity text explicitly states. Do NOT \
infer additional specificity (such as laterality, acuity, type, anatomical \
sub-site, or episode of care) from surrounding context or clinical reasoning \
unless the entity text itself contains that detail.
3. When the entity is a general or abbreviated term (e.g. "cardiac arrest", \
"respiratory failure", "diabetes"), prefer the broader or "unspecified" code \
variant over a more specific sub-code.
4. Return your answer as a JSON object with exactly these keys:
   - "Reason": a short explanation for your choice
   - "Entity": the input entity text (unchanged)
   - "LLM_chosen_code": the ICD-10 code you selected
5. Return ONLY the JSON object, no other text."""

# Appended to SYSTEM_PROMPT only when a standards block is in the prompt.
STANDARDS_RULE = """
6. Canadian Coding Standards (CIHI) for some candidates may follow the candidate \
list. Apply a direction only when the entity and note meet its conditions, and \
ignore it otherwise. The standards never add codes: you must still pick from the \
candidate list."""

USER_PROMPT = """\
Given the following clinical note:
{note_text}

And the entity: "{entity}"

Assign one most appropriate ICD-10 code from the following candidates:
{candidates_json}"""

# RealData: the same prompt without the clinical-note section (run.use_note: false).
USER_PROMPT_NO_NOTE = """\
Given the entity: "{entity}"

Assign one most appropriate ICD-10 code from the following candidates:
{candidates_json}"""

CODE_RE = re.compile(r"[A-Z]\d{2}(?:\.\d{1,4})?")


def to_dotted(code: str) -> str:
    code = (code or "").strip()
    if not code or "." in code:
        return code
    return code if len(code) <= 3 else code[:3] + "." + code[3:]


def parse_pick(raw: str) -> str:
    """Extract the chosen code from the model's final text (JSON or bare code)."""
    raw = (raw or "").strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL)
    body = fence.group(1).strip() if fence else raw
    try:
        return str(json.loads(body).get("LLM_chosen_code", "")).strip()
    except json.JSONDecodeError:
        m = CODE_RE.search(raw)
        return m.group(0) if m else ""


def cache_key(note_id, entity_text, source_method, k, codes, standards="") -> str:
    """Keyed on everything that determines the pick, so a changed candidate set
    (e.g. re-retrieval) invalidates the entry. A non-empty standards block adds
    its hash, so editing the standards re-picks while an entity without any keeps
    its baseline key. Clear the cache if the model or base prompt changes, since
    those are not part of the key."""
    key = f"{note_id}|{entity_text}|{source_method}|{k}|{','.join(codes)}"
    if standards:
        key += "|cs:" + hashlib.sha1(standards.encode("utf-8")).hexdigest()[:12]
    return key


def build_prompts(entity_text, candidates, note_text, max_note_chars, standards="", use_note=True):
    """(system, user) prompts; without standards they are the baseline prompts."""
    if use_note:
        user_prompt = USER_PROMPT.format(
            note_text=(note_text or "")[:max_note_chars],
            entity=entity_text,
            candidates_json=json.dumps(candidates, indent=2),
        )
    else:   # RealData: no clinical note in the prompt
        user_prompt = USER_PROMPT_NO_NOTE.format(
            entity=entity_text,
            candidates_json=json.dumps(candidates, indent=2),
        )
    if not standards:
        return SYSTEM_PROMPT, user_prompt
    return SYSTEM_PROMPT + STANDARDS_RULE, user_prompt + "\n\n" + standards


def call_picker(llm, entity_text, candidates, note_text, max_note_chars, max_new_tokens,
                standards="", use_note=True):
    system_prompt, user_prompt = build_prompts(entity_text, candidates, note_text,
                                               max_note_chars, standards, use_note)
    out = llm.generate(
        user_prompt,
        system_prompt=system_prompt,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
    )
    chosen = to_dotted(parse_pick(out))
    # RealData: an empty pick is no_answer; a pick outside the candidate list is kept and marked not_in_list
    shown = {dotless(c["code"]) for c in candidates}
    status = "no_answer" if not chosen else ("ok" if dotless(chosen) in shown else "not_in_list")
    return {"chosen": chosen, "raw": (out or "").strip()[:2000], "status": status}


def call_picker_batch(llm, jobs, max_note_chars, max_new_tokens, use_note=True):
    """RealData: call_picker for several (entity_text, candidates, note_text, standards) jobs in one generate call."""
    prompts = [build_prompts(e, c, n, max_note_chars, s, use_note) for e, c, n, s in jobs]
    outs = llm.generate_batch([u for _, u in prompts], [s for s, _ in prompts], max_new_tokens=max_new_tokens)
    results = []
    for (_, candidates, _, _), out in zip(jobs, outs):
        chosen = to_dotted(parse_pick(out))
        shown = {dotless(c["code"]) for c in candidates}
        status = "no_answer" if not chosen else ("ok" if dotless(chosen) in shown else "not_in_list")
        results.append({"chosen": chosen, "raw": (out or "").strip()[:2000], "status": status})
    return results


BACKENDS = ("transformers", "unsloth")
STOP_TOKENS = ("generation_config", "tokenizer_eos")   # RealData: see LocalLLM.generate


def load_llm(backend, model_name, device, load_in_8bit, load_in_4bit, max_seq_length,
             enable_thinking, stop_tokens="generation_config"):
    """Load the selected backend when inference begins."""
    if backend == "unsloth":
        return UnslothLLM(model_name=model_name, max_seq_length=max_seq_length,
                          load_in_4bit=load_in_4bit, load_in_8bit=load_in_8bit,
                          enable_thinking=enable_thinking)
    if backend == "transformers":
        return LocalLLM(model_name=model_name, device=device,
                        load_in_8bit=load_in_8bit, load_in_4bit=load_in_4bit, stop_tokens=stop_tokens)
    raise ValueError(f"unknown picker backend {backend!r}; expected one of {BACKENDS}")


def dotless(code: str) -> str:
    return (code or "").upper().replace(".", "")


def prediction_row(entity: dict, top_k: list[str], chosen: str, n_standards: int,
                   status: str = "ok", raw: str = "") -> dict:
    return {
        "entity_id": entity["entity_id"],   # RealData: entity_id, status, raw added
        "status": status,
        "raw": raw,
        "note_id": entity["note_id"],
        "start": entity["start"],
        "end": entity["end"],
        "entity_text": entity["entity_text"],
        "icd_code": entity["ground_truth_codes"][0],
        "llm_icd_code": dotless(chosen),
        "top_k": [dotless(c) for c in top_k],
        "n_standards": n_standards,
    }


# Model backends

class LocalLLM:
    """
    Local LLM inference wrapper for RAG pipeline.
    """

    def __init__(
        self,
        model_name: str = "openai/gpt-oss-20b",
        device: str = "auto",
        torch_dtype: Optional[torch.dtype] = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        stop_tokens: str = "generation_config",
    ):
        """
        Initialize the local LLM.

        Args:
            model_name: HuggingFace model identifier or local path
                Options:
                - "openai/gpt-oss-20b" (20B params, ~40GB GPU memory) - DEFAULT
                - "EleutherAI/gpt-neox-20b" (20B params, ~40GB GPU memory)
                - "EleutherAI/gpt-j-6b" (6B params, ~12GB GPU memory)
                - "mistralai/Mistral-7B-Instruct-v0.2" (7B params)
            device: Device to load model on ('auto', 'cuda', 'cpu', 'mps')
            torch_dtype: Override default dtype (None = auto-detect)
            load_in_8bit: Use 8-bit quantization (reduces memory)
            load_in_4bit: Use 4-bit quantization (reduces memory more)
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        warnings.filterwarnings("ignore", category=UserWarning)
        if stop_tokens not in STOP_TOKENS:
            raise ValueError(f"stop_tokens must be one of {STOP_TOKENS}, got {stop_tokens!r}")
        self.stop_tokens = stop_tokens
        self.model_name = model_name
        self.device_setting = device

        print(f"[LocalLLM] Loading model: {model_name}")
        print(f"[LocalLLM] Device setting: {device}")

        # Determine dtype
        # Use "auto" to let the model use its native dtype (recommended for gpt-oss-20b which uses bfloat16)
        if torch_dtype is None:
            torch_dtype = "auto"  # Let HuggingFace pick the best dtype

        self.torch_dtype = torch_dtype
        print(f"[LocalLLM] Using dtype: {torch_dtype}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model with appropriate settings
        load_kwargs = {
            "torch_dtype": torch_dtype,
            "low_cpu_mem_usage": True,
            "device_map": device,
        }

        # Check if model is already quantized (e.g., gpt-oss uses MXFP4)
        # In that case, skip BitsAndBytes quantization
        is_prequantized_model = any(q in model_name.lower() for q in ['gpt-oss'])

        # Add quantization if requested (only for non-prequantized models)
        if load_in_8bit or load_in_4bit:
            if is_prequantized_model:
                print(f"[LocalLLM] WARNING: {model_name} is already quantized (MXFP4).")
                print("[LocalLLM] Skipping BitsAndBytes quantization - using native quantization.")
            else:
                if load_in_8bit:
                    load_kwargs["load_in_8bit"] = True
                    load_kwargs["device_map"] = "auto"
                    print("[LocalLLM] Using 8-bit quantization")
                elif load_in_4bit:
                    load_kwargs["load_in_4bit"] = True
                    load_kwargs["device_map"] = "auto"
                    print("[LocalLLM] Using 4-bit quantization")

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **load_kwargs
        )

        # Get actual device
        if hasattr(self.model, 'device'):
            self.device = self.model.device
        else:
            self.device = next(self.model.parameters()).device

        print(f"[LocalLLM] Model loaded successfully on {self.device}")

        # Initialize output parser for this model
        self.output_parser = get_output_parser(model_name)
        print(f"[LocalLLM] Using output parser: {self.output_parser.name}")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 50,
        do_sample: bool = False,
        use_chat_template: bool = True,
        parse_output: bool = True,
        return_raw: bool = False
    ) -> str:
        """
        Generate text using the model.

        Args:
            prompt: User input prompt
            system_prompt: Optional system prompt (prepended to input)
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0 = greedy)
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            do_sample: Whether to sample (False = greedy decoding)
            use_chat_template: Whether to use chat template if available
            parse_output: Whether to apply the output parser to remove thinking tokens (default: True)
            return_raw: If True, return tuple of (parsed_output, raw_output) for debugging

        Returns:
            Generated text string (or tuple if return_raw=True)
        """
        import torch

        # Try to use chat template if available (for chat models like gpt-oss-20b)
        if use_chat_template and hasattr(self.tokenizer, 'apply_chat_template'):
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            try:
                inputs = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            except Exception:
                # Fall back to simple concatenation if chat template fails
                use_chat_template = False

        if not use_chat_template or not hasattr(self.tokenizer, 'apply_chat_template'):
            # Construct full prompt (fallback for non-chat models)
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
            else:
                full_prompt = prompt

            # Tokenize
            inputs = self.tokenizer(
                full_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=4000  # Support longer context for gpt-oss-20b
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Set sampling parameters
        # RealData: "generation_config" stops at all of the model's end tokens (Gemma 4: <eos>=1, <turn|>=106, 50);
        # "tokenizer_eos" is the original: only the tokenizer's eos, so Gemma 4 does not stop at the end of its turn.
        eos = (self.model.generation_config.eos_token_id
               if self.stop_tokens == "generation_config" and self.model.generation_config.eos_token_id is not None
               else self.tokenizer.eos_token_id)
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": eos,
        }

        # Temperature 0 means greedy decoding
        if temperature == 0 or not do_sample:
            gen_kwargs["do_sample"] = False
        else:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p
            gen_kwargs["top_k"] = top_k

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **gen_kwargs
            )

        # Decode only the new tokens (excluding the prompt)
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]

        # Decode WITH special tokens to preserve Harmony/thinking markers for parsing
        raw_text_with_markers = self.tokenizer.decode(generated_tokens, skip_special_tokens=False).strip()

        # Apply output parser to extract final response
        if parse_output:
            # RealData: original_input=None, so a reply the parser cannot read gives "" (no_answer); with the prompt
            # here, parse_pick took the first code in the prompt text (usually candidate 1) as the pick
            parsed_text = self.output_parser.extract_final_response(
                raw_output=raw_text_with_markers,
                original_input=None,
                tokenizer=self.tokenizer,
                tokens=generated_tokens.tolist()
            )
        else:
            # If not parsing, strip special tokens for clean output
            parsed_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        if return_raw:
            return parsed_text, raw_text_with_markers
        return parsed_text

    def generate_batch(self, prompts, system_prompts, max_new_tokens: int = 1024):
        """RealData: greedy generate() for several prompts in one call (model.batch_size > 1). Prompts are left-padded;
        each reply is cut after its first end token, so it decodes and parses like a generate() reply."""
        import torch

        convs = [([{"role": "system", "content": s}] if s else []) + [{"role": "user", "content": p}]
                 for p, s in zip(prompts, system_prompts)]
        side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        try:
            inputs = self.tokenizer.apply_chat_template(convs, add_generation_prompt=True, tokenize=True,
                                                        return_dict=True, return_tensors="pt", padding=True)
        finally:
            self.tokenizer.padding_side = side
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        eos = (self.model.generation_config.eos_token_id
               if self.stop_tokens == "generation_config" and self.model.generation_config.eos_token_id is not None
               else self.tokenizer.eos_token_id)
        stop = set(eos) if isinstance(eos, (list, tuple)) else {eos}
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                          pad_token_id=self.tokenizer.eos_token_id, eos_token_id=eos)
        texts = []
        for row in outputs[:, inputs["input_ids"].shape[1]:].tolist():
            cut = next((i + 1 for i, t in enumerate(row) if t in stop), len(row))   # through the first end token
            raw = self.tokenizer.decode(row[:cut], skip_special_tokens=False).strip()
            texts.append(self.output_parser.extract_final_response(raw_output=raw, original_input=None,
                                                                   tokenizer=self.tokenizer, tokens=row[:cut]))
        return texts


class UnslothLLM:
    """Local LLM inference through unsloth's FastModel."""

    def __init__(
        self,
        model_name: str,
        max_seq_length: int = 8192,
        load_in_4bit: bool = True,
        load_in_8bit: bool = False,
        enable_thinking: bool = False,
    ):
        """
        Args:
            model_name: HuggingFace model identifier or local path
                (e.g. "unsloth/gemma-4-31B-it").
            max_seq_length: Context length unsloth allocates for (prompt + output).
            load_in_4bit / load_in_8bit: Quantized loading (4-bit ~18GB for 31B).
            enable_thinking: Passed to the chat template; Gemma 4 reasons in a
                thought channel when True. Off by default, as in the template.
        """
        from unsloth import FastModel

        self.model_name = model_name
        self.enable_thinking = enable_thinking

        print(f"[UnslothLLM] Loading model: {model_name} "
              f"(4bit={load_in_4bit}, 8bit={load_in_8bit}, max_seq_length={max_seq_length})")
        self.model, self.tokenizer = FastModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
        )
        FastModel.for_inference(self.model)

        self.output_parser = get_output_parser(model_name)
        print(f"[UnslothLLM] Model loaded on {self.model.device}; "
              f"output parser: {self.output_parser.name}")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 0,
        parse_output: bool = True,
        return_raw: bool = False,
    ) -> str:
        """Generate one completion. temperature 0 = greedy decoding."""
        import torch

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
        messages.append({"role": "user", "content": [{"type": "text", "text": prompt}]})

        inputs = self.tokenizer.apply_chat_template(
            [messages],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=False,
            enable_thinking=self.enable_thinking,
        ).to(self.model.device)

        gen_kwargs = {"max_new_tokens": max_new_tokens, "use_cache": True}
        if temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            gen_kwargs["do_sample"] = False

        with torch.inference_mode():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        generated = outputs[:, inputs["input_ids"].shape[1]:]
        # Gemma 4's channel markers survive skip_special_tokens (as in the PEFT client).
        raw = self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()

        text = (self.output_parser.extract_final_response(raw_output=raw, original_input=None)   # RealData: see LocalLLM
                if parse_output else raw)
        if return_raw:
            return text, raw
        return text


# Model output parsers

class BaseOutputParser(ABC):
    """Abstract base class for model output parsers."""

    @abstractmethod
    def extract_final_response(
        self,
        raw_output: str,
        original_input: Optional[str] = None,
        tokenizer: Optional[Any] = None,
        tokens: Optional[List[int]] = None
    ) -> str:
        """
        Extract the final response from model output, removing any thinking/reasoning tokens.

        Args:
            raw_output: The raw text output from the model
            original_input: The original input/entity (for fallback)
            tokenizer: The tokenizer used (optional, for token-level parsing)
            tokens: The raw output tokens (optional, for token-level parsing)

        Returns:
            The cleaned final response
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the parser name for logging."""
        pass


class HarmonyParser(BaseOutputParser):
    """
    Parser for gpt-oss models using OpenAI Harmony format.

    Harmony uses channels to separate different types of content:
    - analysis: Chain-of-thought reasoning (internal, not shown to users)
    - commentary: Tool calls and preambles
    - final: The actual answer for the user
    """

    def __init__(self):
        from openai_harmony import HarmonyEncodingName, Role, load_harmony_encoding

        self.assistant_role = Role.ASSISTANT
        try:
            self.encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        except Exception as e:
            print(f"[HarmonyParser] vocab download failed ({e}); "
                  f"falling back to text-based parsing only")
            self.encoding = None

    @property
    def name(self) -> str:
        return "HarmonyParser"

    def extract_final_response(
        self,
        raw_output: str,
        original_input: Optional[str] = None,
        tokenizer: Optional[Any] = None,
        tokens: Optional[List[int]] = None
    ) -> str:
        """
        Extract the final channel content from Harmony-formatted output.

        If tokens are provided, uses token-level parsing for accuracy.
        Otherwise, falls back to text-based parsing.
        """
        # Prefer token-level parsing if tokens are available and encoding loaded
        if tokens is not None and self.encoding is not None:
            try:
                entries = self.encoding.parse_messages_from_completion_tokens(
                    tokens,
                    self.assistant_role
                )

                # Extract content from 'final' channel
                final_contents = []
                for message in entries:
                    msg_dict = message.to_dict()
                    if msg_dict.get('channel') == 'final':
                        content = msg_dict.get('content', '')
                        if content:
                            final_contents.append(content)

                if final_contents:
                    return ' '.join(final_contents).strip()
            except Exception:
                # Fall through to text-based parsing
                pass

        # Text-based fallback: look for channel markers in raw text
        # Format: <|channel|>final<|message|>content
        if '<|channel|>final<|message|>' in raw_output:
            parts = raw_output.split('<|channel|>final<|message|>')
            if len(parts) > 1:
                final_part = parts[-1]
                # Remove any trailing special tokens
                for end_marker in ['<|end|>', '<|return|>', '<|im_end|>']:
                    if end_marker in final_part:
                        final_part = final_part.split(end_marker)[0]
                return final_part.strip()

        # If no channel markers found, return cleaned raw output
        return self._clean_harmony_tokens(raw_output, original_input)

    def _clean_harmony_tokens(self, text: str, original_input: Optional[str] = None) -> str:
        """
        Handle Harmony-formatted text when no <|channel|>final<|message|> was found.

        Two cases:
        - Harmony markers present but no final channel: model never closed its
          reasoning (e.g. runaway loop in analysis). Return empty so the caller
          falls back to the original entity.
        - No Harmony markers at all: skip_special_tokens may have stripped them,
          treat as plain text via DefaultParser.
        """
        harmony_markers = ['<|channel|>', '<|message|>', '<|start|>', '<|end|>']
        has_harmony_format = any(marker in text for marker in harmony_markers)

        if has_harmony_format:
            # Reasoning never reached the final channel — degenerate output.
            return ""

        # No Harmony markers found - use DefaultParser logic as fallback
        default_parser = DefaultParser()
        return default_parser.extract_final_response(text, original_input=original_input)


class DeepSeekParser(BaseOutputParser):
    """
    Parser for DeepSeek R1 models.

    DeepSeek R1 uses <think>...</think> tags for chain-of-thought reasoning.
    The final answer appears after the </think> tag.
    Also handles DeepSeek special tokens like <｜end_of_sentence｜>.
    """

    # Regex pattern to match thinking blocks
    THINK_PATTERN = re.compile(r'<think>.*?</think>', re.DOTALL | re.IGNORECASE)

    # DeepSeek special tokens to strip from final output
    # Note: DeepSeek uses full-width vertical bars ｜ (U+FF5C) in some tokens
    DEEPSEEK_SPECIAL_TOKENS = [
        '<｜end_of_sentence｜>',  # Full-width bars
        '<|end_of_sentence|>',   # Regular bars (fallback)
        '<｜end▁of▁sentence｜>', # Alternative format
        '<｜begin_of_sentence｜>',
        '<|begin_of_sentence|>',
        '<｜end｜>',
        '<|end|>',
        '<｜endoftext｜>',
        '<|endoftext|>',
    ]

    @property
    def name(self) -> str:
        return "DeepSeekParser"

    def _clean_deepseek_tokens(self, text: str) -> str:
        """Remove DeepSeek-specific special tokens from text."""
        result = text
        for token in self.DEEPSEEK_SPECIAL_TOKENS:
            result = result.replace(token, '')
        return result.strip()

    def extract_final_response(
        self,
        raw_output: str,
        original_input: Optional[str] = None,
        tokenizer: Optional[Any] = None,
        tokens: Optional[List[int]] = None
    ) -> str:
        """Extract content after </think> tag, or remove all <think> blocks."""

        # Method 1: If </think> exists, take everything after the last one
        if '</think>' in raw_output.lower():
            # Find the last </think> tag (case-insensitive)
            lower_output = raw_output.lower()
            last_think_end = lower_output.rfind('</think>')
            if last_think_end != -1:
                result = raw_output[last_think_end + len('</think>'):].strip()
                result = self._clean_deepseek_tokens(result)
                if result:
                    return result

        # Method 2: Remove all <think>...</think> blocks
        result = self.THINK_PATTERN.sub('', raw_output).strip()
        result = self._clean_deepseek_tokens(result)

        if result:
            return result

        # Fallback: clean the raw output
        result = self._clean_deepseek_tokens(raw_output)
        return result if result else (original_input or "")


class QwenParser(BaseOutputParser):
    """
    Parser for Qwen thinking models (QwQ, Qwen3, etc.).

    Qwen uses <|start_thinking|>...<|end_thinking|> tags for reasoning.
    The final answer appears after the <|end_thinking|> tag.
    Also handles Qwen special tokens like <|im_end|>, <|im_start|>, etc.
    """

    # Regex pattern to match thinking blocks
    THINK_PATTERN = re.compile(
        r'<\|start_thinking\|>.*?<\|end_thinking\|>',
        re.DOTALL | re.IGNORECASE
    )

    # Qwen special tokens to strip from final output
    QWEN_SPECIAL_TOKENS = [
        '<|im_end|>',
        '<|im_start|>',
        '<|endoftext|>',
        '<|end|>',
        'assistant',  # Sometimes appears after special tokens
    ]

    @property
    def name(self) -> str:
        return "QwenParser"

    def _clean_qwen_tokens(self, text: str) -> str:
        """Remove Qwen-specific special tokens from text."""
        result = text
        for token in self.QWEN_SPECIAL_TOKENS:
            result = result.replace(token, '')
        return result.strip()

    def extract_final_response(
        self,
        raw_output: str,
        original_input: Optional[str] = None,
        tokenizer: Optional[Any] = None,
        tokens: Optional[List[int]] = None
    ) -> str:
        """Extract content after <|end_thinking|> tag and clean Qwen tokens."""

        # Method 1: If end tag exists, take everything after it
        end_marker = '<|end_thinking|>'
        if end_marker.lower() in raw_output.lower():
            lower_output = raw_output.lower()
            last_end = lower_output.rfind(end_marker.lower())
            if last_end != -1:
                result = raw_output[last_end + len(end_marker):].strip()
                result = self._clean_qwen_tokens(result)
                if result:
                    return result

        # Method 2: Remove all thinking blocks
        result = self.THINK_PATTERN.sub('', raw_output).strip()
        result = self._clean_qwen_tokens(result)

        if result:
            return result

        # Fallback: clean the raw output
        result = self._clean_qwen_tokens(raw_output)
        return result if result else (original_input or "")


class GemmaParser(BaseOutputParser):
    """
    Parser for Gemma 4 models.

    Gemma 4 separates reasoning from the answer with channel markers rather than
    hidden reasoning tokens (see https://unsloth.ai/docs/models/gemma-4):

        <|channel>thought
        ...internal reasoning...
        <channel|>
        ...final answer...

    Thinking is enabled by a <|think|> token in the system prompt; when disabled,
    larger models may still emit an empty thought block before the answer. The
    final answer is everything after the thought-closing <channel|> marker, with
    turn/end-of-text tokens stripped.

    The literal markers below come from the unsloth Gemma 4 docs; if the released
    tokenizer differs, update GEMMA_SPECIAL_TOKENS / THINK_PATTERN accordingly.
    """

    # Marker that closes the thought channel; the final answer follows it.
    THOUGHT_END = '<channel|>'

    # Matches a whole thought block (open ... close), including an empty one.
    THINK_PATTERN = re.compile(
        r'<\|channel>thought.*?<channel\|>',
        re.DOTALL | re.IGNORECASE,
    )

    # Gemma special / turn tokens to strip from the final output.
    GEMMA_SPECIAL_TOKENS = [
        '<|channel>thought',
        '<channel|>',
        '<|think|>',
        '<|turn>',
        '<turn|>',
        '<start_of_turn>',
        '<end_of_turn>',
        '<end_of_text>',
        '<eos>',
        '<bos>',
        '<pad>',
    ]

    @property
    def name(self) -> str:
        return "GemmaParser"

    def _clean_gemma_tokens(self, text: str) -> str:
        """Remove Gemma-specific channel/turn tokens from text."""
        result = text
        for token in self.GEMMA_SPECIAL_TOKENS:
            result = result.replace(token, '')
        return result.strip()

    def extract_final_response(
        self,
        raw_output: str,
        original_input: Optional[str] = None,
        tokenizer: Optional[Any] = None,
        tokens: Optional[List[int]] = None
    ) -> str:
        """Extract content after the thought-closing <channel|> marker."""

        lower_output = raw_output.lower()
        has_open = '<|channel>thought' in lower_output
        has_close = self.THOUGHT_END.lower() in lower_output

        # Case 1: thought block closed — the answer follows the last close marker.
        if has_close:
            last_end = lower_output.rfind(self.THOUGHT_END.lower())
            result = self._clean_gemma_tokens(raw_output[last_end + len(self.THOUGHT_END):])
            if result:
                return result
            # Nothing after the close — remove the block and keep whatever remains.
            result = self._clean_gemma_tokens(self.THINK_PATTERN.sub('', raw_output))
            return result if result else (original_input or "")

        # Case 2: thought opened but never closed — degenerate runaway reasoning.
        # Return empty so the caller falls back to the original entity (mirrors
        # HarmonyParser's handling of an unterminated reasoning channel).
        if has_open:
            return original_input or ""

        # Case 3: no channel markers — plain output; strip stray tokens.
        result = self._clean_gemma_tokens(raw_output)
        return result if result else (original_input or "")


class DefaultParser(BaseOutputParser):
    """
    Default heuristic-based parser for models without special thinking token formats.

    This parser uses keyword-based detection to identify reasoning patterns
    and attempts to extract the final answer using common markers.

    This preserves the original logic from extract_final_answer() for backward compatibility.
    """

    # Markers that indicate the output contains reasoning/chain-of-thought
    REASONING_MARKERS = [
        'analysis', 'we need to', 'the entity', 'should be expanded',
        'let me', 'let\'s', 'first,', 'step 1', 'thinking'
    ]

    # Markers that indicate the final answer follows
    FINAL_MARKERS = [
        'final:', 'final answer:', 'expanded entity:', 'result:', 'output:',
        'so:', 'therefore:', 'answer:', '→', '->', 'hence:', 'thus:'
    ]

    @property
    def name(self) -> str:
        return "DefaultParser"

    def extract_final_response(
        self,
        raw_output: str,
        original_input: Optional[str] = None,
        tokenizer: Optional[Any] = None,
        tokens: Optional[List[int]] = None
    ) -> str:
        """
        Extract the final answer using heuristic-based parsing.

        This method preserves the original logic from evaluate_llm_expansion.py
        for backward compatibility.
        """
        response = raw_output.strip()

        if not response:
            return original_input if original_input else ""

        # Check if response starts with reasoning markers
        is_reasoning = any(
            response.lower().startswith(m) for m in self.REASONING_MARKERS
        )

        if is_reasoning:
            # Try to find final answer after common markers
            for marker in self.FINAL_MARKERS:
                if marker in response.lower():
                    idx = response.lower().rfind(marker)
                    extracted = response[idx + len(marker):].strip()
                    # Get first line/phrase
                    extracted = extracted.split('\n')[0].strip()
                    extracted = extracted.strip('"\'.,')
                    if extracted and len(extracted) > 2:
                        return extracted

            # If no marker found, try to get the last quoted phrase
            quotes = re.findall(r'"([^"]+)"', response)
            if quotes:
                # Return the last quoted phrase that looks like a valid term
                for q in reversed(quotes):
                    if len(q) > 2 and not q.lower().startswith('we '):
                        return q.strip()

            # Last resort: return original input if we can't parse
            if original_input:
                return original_input

        # Not reasoning output, just clean it up
        # Get first line only
        response = response.split('\n')[0].strip()
        response = response.strip('"\'')

        return response if response else (original_input or "")


# Model family detection patterns
MODEL_FAMILY_PATTERNS: Dict[str, List[str]] = {
    'harmony': ['gpt-oss', 'openai/gpt-oss'],
    'deepseek': ['deepseek', 'deepseek-r1', 'deepseek-reasoner'],
    'qwen': ['qwen', 'qwq', 'qwen2'],
    # Gemma 4 uses channel markers + a thinking tag (see GemmaParser).
    'gemma': ['gemma'],
    # Models that typically don't have thinking tokens (use default parser)
    'default': ['llama', 'mistral', 'phi', 'falcon', 'mpt'],
}


def detect_model_family(model_name: str) -> str:
    """
    Detect the model family based on the model name.

    Args:
        model_name: The model name or HuggingFace model ID

    Returns:
        The detected model family name ('harmony', 'deepseek', 'qwen', or 'default')
    """
    model_lower = model_name.lower()

    for family, patterns in MODEL_FAMILY_PATTERNS.items():
        if any(pattern in model_lower for pattern in patterns):
            return family

    return 'default'


def get_output_parser(model_name: str) -> BaseOutputParser:
    """
    Get the appropriate output parser for the given model.

    Args:
        model_name: The model name or HuggingFace model ID

    Returns:
        An instance of the appropriate parser for the model

    Example:
        parser = get_output_parser("openai/gpt-oss-20b")
        final_answer = parser.extract_final_response(raw_output)
    """
    family = detect_model_family(model_name)

    if family == 'harmony':
        try:
            return HarmonyParser()
        except ImportError:
            print(
                "[Warning] openai-harmony not installed. "
                "Using DefaultParser for gpt-oss model. "
                "For best results, install with: pip install openai-harmony"
            )
            return DefaultParser()

    elif family == 'deepseek':
        return DeepSeekParser()

    elif family == 'qwen':
        return QwenParser()

    elif family == 'gemma':
        return GemmaParser()

    else:
        return DefaultParser()


# Optional CIHI coding standards

# Most specific first: under the max_chars budget, specific directions win.
LEVEL_ORDER = ["code", "subcategory", "category", "block", "chapter"]

HEADER = ("Canadian Coding Standards (CIHI) attached to these candidates. "
          "Each direction lists the candidate codes it applies to:\n\n")


def norm_code(code: str) -> str:
    """'E11.63' -> 'E1163' (the build script writes prefixes undotted)."""
    return re.sub(r"[^A-Z0-9]", "", (code or "").upper())


def load_standards(path: Path, levels) -> list:
    unknown = set(levels) - set(LEVEL_ORDER)
    if unknown:
        raise ValueError(f"unknown coding-standard levels {sorted(unknown)}; "
                         f"expected a subset of {LEVEL_ORDER}")
    keep = set(levels)
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return [r for r in rows if r["level"] in keep]


def applies(row: dict, code: str) -> bool:
    c = norm_code(code)
    if row.get("prefix"):
        return c.startswith(row["prefix"])
    return row["range_lo"] <= c[:3] <= row["range_hi"]


def standards_block(codes, rows, max_chars: int) -> tuple[str, int, int]:
    """Render the directions that apply to any of `codes`.

    Returns (block, n_included, n_dropped); block is "" when nothing applies.
    Directions are ordered most-specific level first, then by the rank of the
    first candidate they apply to; any that would overflow max_chars are dropped.
    """
    hits = {}
    for code in codes:
        for r in rows:
            if not applies(r, code):
                continue
            h = hits.setdefault(r["direction"], {"title": r["title"], "codes": [],
                                                 "rank": LEVEL_ORDER.index(r["level"])})
            h["rank"] = min(h["rank"], LEVEL_ORDER.index(r["level"]))
            if code not in h["codes"]:
                h["codes"].append(code)

    entries, used, dropped = [], len(HEADER), 0
    for direction, h in sorted(hits.items(), key=lambda kv: kv[1]["rank"]):
        entry = (f"[{len(entries) + 1}] {h['title']} (applies to: {', '.join(h['codes'])})\n"
                 f"{direction.strip()}")
        if used + len(entry) > max_chars:
            dropped += 1
            continue
        entries.append(entry)
        used += len(entry) + 2
    if not entries:
        return "", 0, dropped
    return HEADER + "\n\n".join(entries), len(entries), dropped


# Standalone input and inference loop

def configured_path(settings, key, directory, *, optional=False):
    value = settings.get(key)
    if optional and value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"picker.{key} must be a path string")
    return (directory / Path(value).expanduser()).resolve()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def normalize_codes(codes, context):
    if not isinstance(codes, list) or any(
        not isinstance(code, str) or not code.strip() for code in codes
    ):
        raise ValueError(f"{context}: expected an array of ICD code strings")
    normalized = [to_dotted(code.upper()) for code in codes]
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{context}: duplicate codes")
    return normalized


def prepare_inputs(document, pool_size, reranked=None):
    """Read notes, descriptions, and candidate lists from the consolidated JSON."""
    if reranked is not None and not isinstance(reranked, dict):
        raise ValueError("reranked JSON must be an object mapping entity_id to an array of codes")

    descriptions = {
        to_dotted(code.upper()): description
        for code, description in document["code_descriptions"].items()
    }
    notes = {}
    entities = []
    seen_entities = set()
    for note in document["patient_notes"]:
        note_id = note["note_id"]
        if note_id in notes:
            raise ValueError(f"duplicate note_id: {note_id}")
        notes[note_id] = note["note_text"]
        for entity in note["entities"]:
            entity_id = entity["entity_id"]
            if entity_id in seen_entities:
                raise ValueError(f"duplicate entity_id: {entity_id}")
            seen_entities.add(entity_id)
            slot = entity["rag_outputs"][f"top_{pool_size}"]
            candidates = normalize_codes(slot["candidate_codes"], entity_id)
            # RealData: a list may be shorter than the pool (an entity whose retriever list holds fewer codes)
            if not 1 <= len(candidates) <= pool_size:
                raise ValueError(f"{entity_id}: expected 1 to {pool_size} candidates, got {len(candidates)}")
            if reranked is not None:
                if entity_id not in reranked:
                    raise ValueError(f"reranked JSON is missing {entity_id}")
                ordered = normalize_codes(reranked[entity_id], entity_id)
                if len(ordered) != len(candidates) or set(ordered) != set(candidates):
                    raise ValueError(f"{entity_id}: reranked codes must reorder the original candidate pool")
                candidates = ordered
            for code in candidates:
                if not descriptions.get(code):
                    raise ValueError(f"{entity_id}: missing description for {code}")
            entities.append({
                "entity_id": entity_id,
                "note_id": note_id,
                "entity_text": entity["entity_text"],
                "start": entity["start"],
                "end": entity["end"],
                "ground_truth_codes": [dotless(entity["ground_truth_code"])],
                "candidate_codes": candidates,
            })

    if not entities:
        raise ValueError("input has no entities")
    if reranked is not None and (extra_ids := reranked.keys() - seen_entities):
        raise ValueError(f"reranked JSON contains an unknown entity_id: {min(extra_ids)}")
    return entities, notes, {code: (description or "").strip() for code, description in descriptions.items()}


def prepare_requests(path, pool_size):
    """RealData: read picker_build.py --format requests (one JSON line per entity: entity_id, entity_text, note_text,
    ground_truth_code, candidates [{code, description}] already in the order to show; note_id, start, end when
    given). Returns what prepare_inputs returns; a line without note_id is its own note (note_id = entity_id)."""
    entities, notes, descriptions, seen = [], {}, {}, set()
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    for row in rows:
        entity_id = row["entity_id"]
        if entity_id in seen:
            raise ValueError(f"duplicate entity_id: {entity_id}")
        seen.add(entity_id)
        note_id = row.get("note_id", entity_id)
        if notes.setdefault(note_id, row.get("note_text") or "") != (row.get("note_text") or ""):
            raise ValueError(f"note_id {note_id} has two different note texts")
        candidates = normalize_codes([c["code"] for c in row["candidates"]], entity_id)
        if not 1 <= len(candidates) <= pool_size:
            raise ValueError(f"{entity_id}: expected 1 to {pool_size} candidates, got {len(candidates)}")
        for code, c in zip(candidates, row["candidates"]):
            text = (c.get("description") or "").strip()
            if not text:
                raise ValueError(f"{entity_id}: missing description for {code}")
            if descriptions.setdefault(code, text) != text:
                raise ValueError(f"{code} has two different descriptions")
        entities.append({
            "entity_id": entity_id,
            "note_id": note_id,
            "entity_text": row["entity_text"],
            "start": row.get("start", 0),
            "end": row.get("end", 0),
            "ground_truth_codes": [dotless(row["ground_truth_code"])],
            "candidate_codes": candidates,
        })
    if not entities:
        raise ValueError("input has no entities")
    return entities, notes, descriptions


def run_picker(inputs, run, model, run_id, standards_path, output, llm=None):
    """Select one code per entity with the pipeline prompt and model backends."""
    entities, notes, descriptions = inputs
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    backend = model.get("backend", "transformers")
    k = run["k"]
    use_note = run.get("use_note", True)   # RealData: false = the prompt without the note section
    standards = run.get("coding_standards") or {}
    std_rows = load_standards(standards_path, standards["levels"]) if standards_path and standards else []
    max_chars = standards.get("max_chars", 6000)

    log.info("Run %s: %d entities, K=%d, backend=%s, model=%s, note in prompt: %s",
             run_id, len(entities), k, backend, model["model_name"], use_note)
    if llm is None:   # RealData: main() passes one loaded model to several runs
        llm = load_llm(backend, model["model_name"], model.get("device", "auto"),
                       model.get("load_in_8bit", False), model.get("load_in_4bit", False),
                       model.get("max_seq_length", 8192), model.get("enable_thinking", False),
                       model.get("stop_tokens", "generation_config"))

    cache = {}
    batch_size = model.get("batch_size", 1)
    if batch_size > 1:   # RealData: answer every distinct prompt in batches first; the loop below then reads the cache
        todo = {}
        for entity in entities:
            candidates = entity["candidate_codes"][:k]
            block = standards_block(candidates, std_rows, max_chars)[0] if std_rows else ""
            key = cache_key(entity["note_id"], entity["entity_text"], RANKING_KEY, k, candidates, block)
            if key not in todo:
                cands = [{"code": code, "description": descriptions.get(code, "")} for code in candidates]
                todo[key] = (entity["entity_text"], cands, notes.get(entity["note_id"], ""), block)
        keys = sorted(todo, key=lambda kk: len(json.dumps(todo[kk])))   # similar lengths together: less padding
        for i in tqdm(range(0, len(keys), batch_size), desc=f"RAG picker [{run_id}] batches of {batch_size}"):
            chunk = keys[i:i + batch_size]
            cache.update(zip(chunk, call_picker_batch(llm, [todo[kk] for kk in chunk], run["max_note_chars"],
                                                      model["max_new_tokens"], use_note)))
    rows = []
    dropped = 0
    for entity in tqdm(entities, desc=f"RAG picker [{run_id}]", disable=batch_size > 1):
        candidates = entity["candidate_codes"][:k]
        block, n_standards, n_dropped = (
            standards_block(candidates, std_rows, max_chars) if std_rows else ("", 0, 0)
        )
        dropped += n_dropped
        key = cache_key(entity["note_id"], entity["entity_text"], RANKING_KEY, k, candidates, block)
        if key not in cache:
            prompt_cands = [{"code": code, "description": descriptions.get(code, "")} for code in candidates]
            cache[key] = call_picker(
                llm, entity["entity_text"], prompt_cands, notes.get(entity["note_id"], ""),
                run["max_note_chars"], model["max_new_tokens"], block, use_note,
            )
        rows.append(prediction_row(entity, candidates, cache[key]["chosen"], n_standards,
                                   cache[key]["status"], cache[key]["raw"]))

    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(output.name + ".tmp")   # RealData: written whole, then renamed (no half-written output)
    with tmp.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    if output.exists():
        raise ValueError(f"output already exists: {output}")
    # RealData: the settings that change the picks (bf16 results depend on the batch size), next to the output
    output.with_name(output.name + ".meta.json").write_text(json.dumps(
        {"model": model, "k": k, "use_note": use_note, "entities": len(rows), "model_calls": len(cache)}, indent=1) + "\n",
        encoding="utf-8")
    tmp.replace(output)
    bad = sum(r["status"] != "ok" for r in rows)
    log.info("Wrote %d predictions in %d model calls -> %s (%d dropped standards directions, %d not ok)",
             len(rows), len(cache), output, dropped, bad)


def load_config(config_path):
    """One --config: (inputs, run, model, run_id, standards, output). Raises on a bad config or input."""
    config_path = config_path.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8-sig"))
    settings = config["picker"]
    directory = config_path.parent
    input_format = settings.get("format", "native")   # RealData: native (consolidated JSON) or requests (JSONL)
    if input_format not in ("native", "requests"):
        raise ValueError(f"picker.format must be native or requests, got {input_format!r}")
    input_path = configured_path(settings, "input", directory)
    reranked_path = configured_path(settings, "reranked", directory, optional=True)
    output = configured_path(settings, "output", directory)
    standards = configured_path(settings, "standards", directory, optional=True)

    run_name = settings["run"]
    model_key = settings["model_key"]
    run = config["rag"]["runs"][run_name]
    model = config["rag"]["models"][model_key]
    k = run["k"]
    pool_size = settings.get("candidate_pool")
    if pool_size is None:
        pool_size = k
    if type(k) is not int or type(pool_size) is not int or not 1 <= k <= pool_size:
        raise ValueError("candidate pool must be at least the run's positive integer K")
    if run.get("coding_standards"):
        if standards is None or not standards.is_file():
            raise ValueError("this run requires picker.standards pointing to the built CIHI JSONL")
    elif standards is not None:
        raise ValueError("the selected run has coding standards disabled")
    if model.get("backend", "transformers") not in BACKENDS:
        raise ValueError(f"picker backend must be one of {BACKENDS}")
    if model.get("stop_tokens", "generation_config") not in STOP_TOKENS:
        raise ValueError(f"model.stop_tokens must be one of {STOP_TOKENS}")
    batch_size = model.get("batch_size", 1)
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("model.batch_size must be a whole number of 1 or more")
    if batch_size > 1 and model.get("backend", "transformers") != "transformers":
        raise ValueError("model.batch_size > 1 needs the transformers backend")
    if input_format == "requests":
        if reranked_path is not None:
            raise ValueError("picker.reranked is for the native format; a requests file holds its own order")
        inputs = prepare_requests(input_path, pool_size)
    else:
        reranked = read_json(reranked_path) if reranked_path is not None else None
        if reranked_path is not None and reranked is None:
            raise ValueError("reranked JSON must be an object, got null")
        inputs = prepare_inputs(read_json(input_path), pool_size, reranked)
    return inputs, run, model, f"{run_name}-{model_key}", standards, output, k, pool_size


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, nargs="+",
                        help="YAML with picker paths and run settings (several: one model load per model setting)")
    parser.add_argument("--validate-only", action="store_true", help="check inputs without loading a model")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="RealData: entities per generate call, overriding model.batch_size (default: the config's, else 1)")
    args = parser.parse_args()
    if args.batch_size is not None and args.batch_size < 1:
        parser.error("--batch-size must be 1 or more")

    jobs = []
    for config_path in args.config:
        try:
            job = load_config(config_path)
        except (AttributeError, ImportError, KeyError, OSError, TypeError, ValueError, yaml.YAMLError) as error:
            parser.error(f"{config_path}: invalid inputs or configuration: {error}")
        inputs, run, model, run_id, standards, output, k, pool_size = job
        if args.batch_size is not None:
            if args.batch_size > 1 and model.get("backend", "transformers") != "transformers":
                parser.error("--batch-size > 1 needs the transformers backend")
            model["batch_size"] = args.batch_size
        job = (inputs, run, model, run_id, standards, output, k, pool_size)
        print(f"{config_path.name}: validated {len(inputs[0])} entities. "
              f"The picker will see {k} of {pool_size} codes per entity.")
        if output.exists():   # RealData: a finished run is kept, not redone
            print(f"{config_path.name}: skipped, output already exists: {output}")
            continue
        jobs.append(job)
    if args.validate_only or not jobs:
        return

    loaded = {}   # RealData: one model load for every config with the same model settings
    for inputs, run, model, run_id, standards, output, k, pool_size in jobs:
        key = json.dumps({x: v for x, v in model.items() if x != "batch_size"}, sort_keys=True)   # same weights, any batch
        if key not in loaded:
            loaded.clear()   # free the previous model first
            loaded[key] = load_llm(model.get("backend", "transformers"), model["model_name"],
                                   model.get("device", "auto"), model.get("load_in_8bit", False),
                                   model.get("load_in_4bit", False), model.get("max_seq_length", 8192),
                                   model.get("enable_thinking", False),
                                   model.get("stop_tokens", "generation_config"))
        run_picker(inputs, run, model, run_id, standards, output, llm=loaded[key])
        print(f"Saved predictions to {output}")


if __name__ == "__main__":
    main()
