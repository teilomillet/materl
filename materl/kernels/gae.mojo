from gpu import block_dim, block_idx, thread_idx
from gpu.host import DeviceContext
from runtime.asyncrt import DeviceContextPtr
from tensor_internal import InputTensor, ManagedTensorSlice, OutputTensor
from utils.index import Index
from dtypes import DType

from arch.helpers import shared_memory, sync_threads
from string import StaticString

alias FP32 = DType.float32


@parameter
fn gae_kernel[
    BLOCK_SIZE: Int,
    SEQ_LEN: Int,
    BATCH_SIZE: Int,
](
    # All tensors are (BATCH_SIZE, SEQ_LEN)
    rewards: ManagedTensorSlice[FP32],
    values: ManagedTensorSlice[FP32],
    completion_masks: ManagedTensorSlice[FP32],
    gamma: FP32,
    lam: FP32,
    # Outputs
    advantages: ManagedTensorSlice[FP32, mut=True],
    returns: ManagedTensorSlice[FP32, mut=True],
) raises:
    # Each block processes one sequence in the batch
    let batch_idx = block_idx.x
    if batch_idx >= BATCH_SIZE {
        return
    }

    # Use shared memory for the single sequence this block is handling
    let smem_rewards = shared_memory[FP32, SEQ_LEN]()
    let smem_values = shared_memory[FP32, SEQ_LEN]()
    let smem_masks = shared_memory[FP32, SEQ_LEN]()

    # Parallel load from global to shared memory
    let tid = thread_idx.x
    for i in range(tid, SEQ_LEN, BLOCK_SIZE) {
        let idx = Index(batch_idx, i)
        smem_rewards[i] = rewards.load[1](idx)
        smem_values[i] = values.load[1](idx)
        smem_masks[i] = completion_masks.load[1](idx)
    }
    sync_threads()

    # Sequential GAE calculation, performed by the first thread in the block
    if tid == 0 {
        var last_advantage: FP32 = 0.0
        for t in range(SEQ_LEN - 1, -1, -1) {
            let mask = smem_masks[t]

            let next_values = smem_values[t + 1] if t < SEQ_LEN - 1 else 0.0
            let delta = smem_rewards[t] + gamma * next_values - smem_values[t]

            # GAE formula: A_t = delta_t + gamma * lambda * A_{t+1}
            # The mask ensures that the advantage resets to zero for padded tokens
            # and at the start of a new sequence within the padding.
            last_advantage = delta + gamma * lam * last_advantage * mask
            let current_advantage = last_advantage

            # Write advantage and return (value target) to global memory
            advantages.store[1](Index(batch_idx, t), current_advantage)
            returns.store[1](Index(batch_idx, t), current_advantage + smem_values[t])
        }
    }
}


from compiler import compiler


@compiler.register("gae")
struct GAE:
    @staticmethod
    fn execute[
        target: StaticString,
    ](
        # Define output tensors first
        advantages: OutputTensor[rank=2],
        returns: OutputTensor[rank=2],
        # Input tensors
        rewards: InputTensor[dtype=FP32, rank=2],
        values: InputTensor[dtype=FP32, rank=2],
        completion_masks: InputTensor[dtype=FP32, rank=2],
        gamma: InputTensor[dtype=FP32, rank=0],
        lam: InputTensor[dtype=FP32, rank=0],
        # MAX context
        ctx: DeviceContextPtr,
    ) raises:
        @parameter
        if target == "gpu":
            let BATCH_SIZE = rewards.dim_size(0)
            let SEQ_LEN = rewards.dim_size(1)

            # This kernel uses one block per sequence in the batch.
            # The number of threads per block is tunable.
            alias BLOCK_THREADS = 128
            let grid = (BATCH_SIZE, 1, 1)
            let block = (BLOCK_THREADS, 1, 1)

            let gamma_scalar = gamma.load[1](Index())
            let lam_scalar = lam.load[1](Index())

            let gpu_ctx = ctx.get_device_context()
            gpu_ctx.enqueue_function[
                gae_kernel[BLOCK_THREADS, SEQ_LEN, BATCH_SIZE]
            ](
                rewards,
                values,
                completion_masks,
                gamma_scalar,
                lam_scalar,
                advantages,
                returns,
                grid_dim=grid,
                block_dim=block,
            )
        else:
            # CPU fallback could be implemented here if needed.
            raise Error("GAE kernel currently only supports GPU.")
