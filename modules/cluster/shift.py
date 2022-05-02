# coding=utf-8
"""
temporal shift / token shift for vision transformers

Rerference:
[1] https://github.com/sallymmx/ActionCLIP/blob/7e734d465d3c241849fd07d9d88d8fc031138f2d/modules/temporal_shift.py

[2] https://github.com/VideoNetworks/TokShift-Transformer/blob/main/vit_models/modeling_tokshift.py#L185

[3] Token Shift Transformer for Video Classification, MM 2021.
"""
import torch


def temporal_shift_wo_cls(x, n_segment, fold_div=8):
	"""
	temporal shift without class embedding
	Args:
		x: [batch_size, sequence_length, D]
	"""
	nt, hw, c = x.size()
	cls_ = x[:, 0:1, :]
	x = x[:, 1:, :]
	n_batch = nt // n_segment
	x = x.contiguous().view(n_batch, n_segment, hw - 1, c)
	fold = c // fold_div

	out = torch.zeros_like(x)
	out[:, :-1, :, :fold] = x[:, 1:, :, :fold]  					# shift left
	out[:, 1:, :, fold: 2 * fold] = x[:, :-1, :, fold: 2 * fold]  	# shift right
	out[:, :, :, 2 * fold:] = x[:, :, :, 2 * fold:]  				# not shift
	out = out.contiguous().view(nt, hw - 1, c)

	out = torch.cat((cls_, out), dim=1).contiguous()

	return out


def token_shift(x, n_segment, fold_div=8):
	"""
	token shift algorithm in
	Token Shift Transformer for Video Classification, MM 2021.
	Only shift the [CLASS] token.
	Args:
		x: [batch_size, sequence_length, D]
	"""
	t = n_segment
	bt, n, c = x.size()
	b = bt // t
	x = x.view(b, t, n, c) 												# B, T, N, C

	fold = c // fold_div
	out  = torch.zeros_like(x)
	# Here only shift [CLASS] token
	out[:, :-1, 0, :fold] = x[:, 1:, 0, :fold] 							# shift left
	out[:, 1:,  0, fold : 2 * fold] = x[:, :-1, 0, fold : 2 * fold]		# shift right

	out[:, :, 1:, :2 * fold] = x[:, :, 1:, :2 * fold]
	out[:, :, :, 2 * fold:] = x[:, :, :, 2 * fold:]

	return out.view(bt, n, c)
