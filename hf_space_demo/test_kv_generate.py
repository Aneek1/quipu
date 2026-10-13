"""Cached vs uncached generation must agree (run from hf_space_demo/, local export folder)."""
import sys, time, torch
torch.set_num_threads(4)
sys.path.insert(0, "../hf_export_moe_chat_distill")
import modeling_quipu_moe as mq
import kv_generate as kv

kv.bind(mq)
model = mq.load("../hf_export_moe_chat_distill").eval()
tok = mq.load_tokenizer("../hf_export_moe_chat_distill")
END = tok.special_id("<|end|>")
stops = {END, tok.eot}

def prompt(text):
    return [tok.special_id("<|user|>"), *tok.encode(text), END, tok.special_id("<|assistant|>")]

# 1) logits: cached prefill + steps == full forward at every position
ids = prompt("Write a Python function that checks whether a string is a palindrome.")
with torch.no_grad():
    full = model(torch.tensor([ids]))
    caches = [None] * len(model.blocks)
    pre = kv.forward_cached(model, torch.tensor([ids[:-5]]), 0, caches)
    steps = [kv.forward_cached(model, torch.tensor([[t]]), len(ids) - 5 + j, caches) for j, t in enumerate(ids[-5:])]
    got = torch.cat([pre] + steps, dim=1)
print("max |logit diff| cached vs full:", (got - full).abs().max().item(), "max |logit|", full.abs().max().item())

# 2) greedy token streams identical, and speed
for text in ["Write a Python function that returns the n-th Fibonacci number.", "Apa itu fotosintesis?",
             "What is photosynthesis? Explain it in two sentences."]:
    ids = prompt(text); res = {}
    for use_cache in (False, True):
        t = time.time()
        out = list(kv.stream(model, ids, 64, stops, temperature=0.0, repetition_penalty=1.15, use_cache=use_cache))
        res[use_cache] = (out, time.time() - t)
    same = res[True][0] == res[False][0]
    print(f"{text[:40]!r}: identical={same} tokens={len(res[True][0])} "
          f"uncached {len(res[False][0]) / res[False][1]:.1f} tok/s, cached {len(res[True][0]) / res[True][1]:.1f} tok/s")
    assert same, (tok.decode(res[True][0]), tok.decode(res[False][0]))
print("OK")
