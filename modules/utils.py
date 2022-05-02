# coding=utf-8
import torch
import logging

logger = logging.getLogger(__name__)


def log_info(info):
    """log for distributed pytorch training"""
    if not torch.distributed.is_initialized() or \
        (torch.distributed.is_initialized() and torch.distributed.get_rank() == 0):
        logging.info(info)


def update_attr(target_name, target_config, target_attr_name, source_config, source_attr_name, default_value=None):
    """"update target_config.target_attr_name = source_config.source_attr_name"""
    if hasattr(source_config, source_attr_name):
        if default_value is None or getattr(source_config, source_attr_name) != default_value:
            setattr(target_config, target_attr_name, getattr(source_config, source_attr_name))
            log_info(source_config, "Set {}.{}: {}.".format(target_name,
                                                            target_attr_name, getattr(target_config, target_attr_name)))
    return target_config


class AllGather(torch.autograd.Function):
    """An autograd function that performs allgather on a tensor."""

    @staticmethod
    def forward(ctx, tensor, args=None):
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        
        output = [torch.empty_like(tensor) for _ in range(world_size)]
        torch.distributed.all_gather(output, tensor)
        ctx.rank = rank
        ctx.batch_size = tensor.shape[0]
        return torch.cat(output, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        return (
            grad_output[ctx.batch_size * ctx.rank : ctx.batch_size * (ctx.rank + 1)],
            None,
        )


def all_gather(tensor):
    """gather a tensor from all devices in distributed training"""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()

        # We gather tensors from all gpus to get more negatives to contrast with.
        gathered_tensor = [torch.zeros_like(tensor) for _ in range(world_size)]
        torch.distributed.all_gather(gathered_tensor, tensor)
        gathered_tensor[rank] = tensor

        # all_tensor = torch.cat(
        #     [tensor]
        #     + gathered_tensor[:rank]
        #     + gathered_tensor[rank + 1 :]
        # )

        return torch.cat(gathered_tensor, dim=0)
