# Prompt History Format

## Turn Structure

`HistoryPromptBuilder` emits natural conversation turns without artificial headers:

```
<|im_start|>system
[instruction prompt]
<|im_end|>
<|im_start|>user
[previous observation - short_term + long_term]
<|im_end|>
<|im_start|>assistant
[previous reasoning/response]
<|im_end|>
<|im_start|>user
[current observation]
<|im_end|>
```

## Key Parameters

- `max_cot_history`: Number of previous reasoning turns to include (default: 1)
- `max_text_history`: Number of previous observations to include long_term text

## Files

- `verl/envs/captioners/prompt_builder/history.py`
- Tests: `test_history_prompt_builder.py`
