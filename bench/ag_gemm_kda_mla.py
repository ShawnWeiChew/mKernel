import torch
import torch.distributed as dist
import math

dist.init_process_group(backend="nccl")
world_size = dist.get_world_size()

GLOBAL_M = [2048, 3072, 3584, 4096, 8192, 16384, 32768]

assert (12288 * 4 + 96) % world_size == 0 and 12288 % world_size == 0, "Shared dimensions cannot be cleanly divided"

LOCAL_K = 7168

# split everything but the head dim, round up to nearest multiple of 16 to fit TK
LOCAL_KDA_N = math.ceil(((12288 * 4 + 96) // world_size + 128) / 16) 

# each rank needs its own MLA kv cache, but the g projection is shared, round up to nearest multiple of 16 to fit TK
LOCAL_MLA_N = math.ceil((576 + 1536 + 12288 // world_size) / 16)