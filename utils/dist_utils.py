# coding=utf-8
import torch
import threading
import torch.nn as nn
import torch.distributed as dist
from torch._utils import ExceptionWrapper


def get_a_var(obj):
    if isinstance(obj, torch.Tensor):
        return obj

    if isinstance(obj, list) or isinstance(obj, tuple):
        for result in map(get_a_var, obj):
            if isinstance(result, torch.Tensor):
                return result
    if isinstance(obj, dict):
        for result in map(get_a_var, obj.items()):
            if isinstance(result, torch.Tensor):
                return result
    return None


def parallel_apply(fct, model, inputs, device_ids):
    modules = nn.parallel.replicate(model, device_ids)
    assert len(modules) == len(inputs)
    lock = threading.Lock()
    results = {}
    grad_enabled = torch.is_grad_enabled()

    def _worker(i, module, input):
        torch.set_grad_enabled(grad_enabled)
        device = get_a_var(input).get_device()
        try:
            with torch.cuda.device(device):
                # this also avoids accidental slicing of `input` if it is a Tensor
                if not isinstance(input, (list, tuple)):
                    input = (input,)
                output = fct(module, *input)
            with lock:
                results[i] = output
        except Exception:
            with lock:
                results[i] = ExceptionWrapper(where="in replica {} on device {}".format(i, device))

    if len(modules) > 1:
        threads = [threading.Thread(target=_worker, args=(i, module, input))
                    for i, (module, input) in enumerate(zip(modules, inputs))]

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    else:
        _worker(0, modules[0], inputs[0])

    outputs = []
    for i in range(len(inputs)):
        output = results[i]
        if isinstance(output, ExceptionWrapper):
            output.reraise()
        outputs.append(output)
    return outputs


def is_dist_avail_and_initialized():
	if not dist.is_available():
		return False
	if not dist.is_initialized():
		return False
	return True


def get_rank():
    """return global rank"""
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_master():
    return (get_rank() == 0 or not torch.cuda.is_available())


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed_mode(args, ngpus_per_node, gpu):
	"""initialize for distributed training"""

	if args.distributed:
		print("INFO: [CUDA] Initialize process group for distributed training")
		global_rank = args.local_rank * ngpus_per_node + gpu
		print("INFO: [CUDA] Use [GPU: {} / Global Rank: {}] for training, "
						"init_method {}, world size {}".format(gpu, global_rank, args.init_method, args.world_size))
		# set device before init process group
		# Ref: https://github.com/pytorch/pytorch/issues/18689
		torch.cuda.set_device(args.gpu)	
		torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.init_method,
												world_size=args.world_size, rank=global_rank)
		torch.distributed.barrier(device_ids=[args.gpu])
		setup_for_distributed(global_rank == 0)

	else:
		args.local_rank = gpu
		global_rank = 0
		print("Use [GPU: {}] for training".format(gpu))

	return global_rank
