import torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from lightbinpack import radix_merge

example_1 = list(range(512)) + list(range(1024, 1280)) + list(range(2048, 4096))
example_2 = list(range(512)) + list(range(1024, 2048))
example_3 = list(range(1024))

example_4 = list(range(4096, 5120)) + list(range(6144, 8192))
example_5 = list(range(4096, 6144))

examples = [[example_1, example_2, example_3], [example_4, example_5]]

_, total_lengths, _, _, index_lists = radix_merge(examples, max_count=48, allow_cross_group_merge=False)
total_length = sum(total_lengths)
index_list = [index + (i << 48) for i, index_list in enumerate(index_lists) for index in index_list]

def generate_doc_sets_ids_mask(offset_to_doc_sets_ids):
    def doc_mask(b, h, q_idx, kv_idx):
        return (offset_to_doc_sets_ids[q_idx] << 16 & offset_to_doc_sets_ids[kv_idx] << 16 == offset_to_doc_sets_ids[q_idx] << 16) & (offset_to_doc_sets_ids[q_idx] >> 48 == offset_to_doc_sets_ids[kv_idx] >> 48) & (q_idx >= kv_idx)
    return doc_mask

flex_attention = torch.compile(flex_attention, fullgraph=True)

qkv = [
    torch.randn(1, 1, total_length, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    for _ in range(3)
]

block_mask_fn = generate_doc_sets_ids_mask(torch.tensor(index_list, device="cuda", dtype=torch.uint64))
block_mask = create_block_mask(block_mask_fn, 1, 1, total_length, total_length, device="cuda", _compile=True)

print("Block mask:")
print(block_mask)

out = flex_attention(*qkv, block_mask=block_mask)
print("Forward pass successful")

grad_out = torch.randn_like(out)
out.backward(grad_out, retain_graph=True)
print("Backward pass successful")
