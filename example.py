import torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from lightbinpack import radix_merge

example_1 = list(range(512)) + list(range(1024, 1280)) + list(range(2048, 4096))
example_2 = list(range(512)) + list(range(1024, 2048))
example_3 = list(range(1024))
examples = [[example_1, example_2, example_3]]

_, total_lengths, _, _, index_lists = radix_merge(examples)

def generate_doc_sets_mask(offset_to_doc_sets):
    def doc_mask(b, h, q_idx, kv_idx):
        return (offset_to_doc_sets[q_idx] & offset_to_doc_sets[kv_idx] == offset_to_doc_sets[q_idx]) & (q_idx >= kv_idx)

    return doc_mask

flex_attention = torch.compile(flex_attention, fullgraph=True)

qkv = [
    torch.randn(1, 1, total_lengths[0], 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    for _ in range(3)
]

block_mask_fn = generate_doc_sets_mask(torch.tensor(index_lists[0], device="cuda", dtype=torch.uint64))
block_mask = create_block_mask(block_mask_fn, 1, 1, total_lengths[0], total_lengths[0], device="cuda", _compile=True)

print("Block mask:")
print(block_mask)

out = flex_attention(*qkv, block_mask=block_mask)
print("Forward pass successful")

grad_out = torch.randn_like(out)
out.backward(grad_out, retain_graph=True)
print("Backward pass successful")
