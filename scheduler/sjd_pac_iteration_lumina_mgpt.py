"""SJD-PAC decoding for Lumina-mGPT.

This module implements the SJD-PAC sampler used to accelerate the
auto-regressive text-to-image generation of Lumina-mGPT. It patches a
``FlexARInferenceSolver`` pipeline through :func:`renew_pipeline_sampler`, which
swaps in the SJD-PAC pipeline, sampler and backbone classes defined below.
"""

import json
import random
from typing import Optional, Tuple, Union

import numpy as np
import torch
from absl import logging
from torch import nn
from transformers import GenerationConfig
from transformers.cache_utils import Cache, StaticCache
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import (
    EosTokenCriteria,
    StoppingCriteriaList,
)
from transformers.generation.utils import (
    GenerateDecoderOnlyOutput,
    GenerateEncoderDecoderOutput,
    GenerateNonBeamOutput,
)
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.utils import is_torchdynamo_compiling

from .logit_processor_3dim import (
    MultiTokensInterleavedTopKLogitsWarper,
    MultiTokensVLLogitsProcessor,
    get_double_cfg_input_ids,
)


def set_seed(seed: int):
    """
    Args:
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch`.
        seed (`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def delete_false_key_value(
    self,
    num_of_false_tokens,
) -> Tuple[torch.Tensor, torch.Tensor]:

    for layer_idx in range(len(self.key_cache)):
        self.key_cache[layer_idx] = self.key_cache[layer_idx][
            ..., :-num_of_false_tokens, :
        ]
        self.value_cache[layer_idx] = self.value_cache[layer_idx][
            ..., :-num_of_false_tokens, :
        ]


def postprocess_cfg_decode(
    model_inputs,
    cfg_half_name_list=[
        "inputs_embeds",
        "input_ids",
        "pixel_values",
    ],
):
    cfg_half_name_list = cfg_half_name_list

    def cfg_half(x):
        return x[: x.shape[0] // 2]

    for name in cfg_half_name_list:
        if (name in model_inputs) and (model_inputs[name] is not None):
            model_inputs[name] = cfg_half(model_inputs[name])

    return model_inputs


def check_is_force_no_cfg(
    input_ids,
    image_start_token_id=None,
    image_end_token_id=None,
):
    if (image_start_token_id is None) or (image_end_token_id is None):
        return False

    num_image_start_tokens = (input_ids[0] == image_start_token_id).sum()
    num_image_end_tokens = (input_ids[0] == image_end_token_id).sum()

    if num_image_start_tokens == num_image_end_tokens:
        return True
    else:
        return False


class SpecEosCriteria(EosTokenCriteria):
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> torch.BoolTensor:
        self.eos_token_id = self.eos_token_id.to(input_ids.device)

        current_length = input_ids.shape[-1]
        diff = current_length - getattr(self, "_last_length", current_length - 1)
        self._last_length = current_length

        is_done = torch.isin(input_ids[:, -diff:], self.eos_token_id).any(dim=-1)
        return is_done


class SJDPACSpeculativeSampler:
    def __init__(
        self,
        generator=None,
        draft_type="jacobian_states",
    ):
        self.draft_token_index_selector = lambda x: x
        if draft_type == "jacobian_states":
            # for jacobi iteration (predict next token)
            self.advanced_token_index_selector = lambda x: x - 1
        else:
            self.advanced_token_index_selector = lambda x: x
        self.generator = generator
        self.image_token_list = [i for i in range(4, 8195 + 1)]
        self.img_start = 4
        self.img_end = 8196
        self.uni_val = 1.0 / (self.img_end - self.img_start)

    def __call__(
        self,
        draft_tokens,
        draft_prob,
        advanced_prob,
    ):
        _, L, V = advanced_prob.shape
        L_t = L - 1

        draft_start = self.draft_token_index_selector(1)
        draft_end = self.draft_token_index_selector(L)
        advanced_start = self.advanced_token_index_selector(1)
        advanced_end = self.advanced_token_index_selector(L)

        p = advanced_prob[:, advanced_start:advanced_end]
        q = draft_prob[:, draft_start:draft_end].repeat(p.shape[0], 1, 1)
        d_tkns = draft_tokens[:, draft_start:draft_end]

        parent_tkn = d_tkns[self.p_idx, self.c_idx]
        q[self.b_idx, self.c_idx, parent_tkn] = 0
        q_child = q[self.u_b_idx, self.u_c_idx]
        q_child.div_(q_child.sum(dim=-1, keepdim=True))
        q[self.u_b_idx, self.u_c_idx] = q_child

        for curr_b, curr_c, prev_b in zip(
            self.seq_curr_b, self.seq_curr_c, self.seq_prev_b
        ):
            p_prev = p[prev_b, curr_c]
            q_prev = q[prev_b, curr_c]
            p_curr = p_prev.sub_(q_prev).clamp_min_(0)
            p_curr.div_(p_curr.sum(dim=-1, keepdim=True))
            p[curr_b, curr_c] = p_curr

        rnd = torch.rand(d_tkns.shape, device=d_tkns.device, generator=self.generator)
        rnd[:, : self.tree_depth - 1] = rnd.gather(0, self.node_group_heads)
        rnd[1:, self.tree_depth - 1] = torch.inf

        tkn_id = d_tkns.clamp_min(0).unsqueeze(-1)
        p_tkn = p.gather(-1, tkn_id).squeeze(-1)
        q_tkn = q.gather(-1, tkn_id).squeeze(-1).mul_(rnd)
        acceptance_map = p_tkn > q_tkn
        accepted_lengths = acceptance_map.cummin(dim=1).values.sum(dim=1)
        acc_len, acc_row = torch.max(accepted_lengths, dim=0)
        acc_len = acc_len.item()
        acc_row = acc_row.item()
        res_len = L_t - acc_len

        target_cols = torch.arange(acc_len, L_t, device=d_tkns.device)
        target_rows = torch.where(target_cols < self.tree_depth, acc_row, 0)
        rej_vector = ~acceptance_map[target_rows, target_cols]

        res_rows = target_rows.clone()
        res_tree_len = self.tree_depth - acc_len - 1
        if res_tree_len > 0:
            msk = rej_vector[:res_tree_len]
            res_rows[:res_tree_len] = torch.where(
                msk,
                self.last_siblings[
                    target_rows[:res_tree_len], target_cols[:res_tree_len]
                ],
                target_rows[:res_tree_len],
            )
        q[1:, self.tree_depth - 1] = 0
        res_p = p[res_rows, target_cols].sub_(q[res_rows, target_cols]).clamp_min_(0)
        res_p[..., 0].add_(1e-20)
        if (
            getattr(self, "_uniform_pad", None) is None
            or self._uniform_pad.shape[0] < L
        ):
            self._uniform_pad = torch.zeros((L, V), dtype=p.dtype, device=p.device)
            self._uniform_pad[:, self.img_start : self.img_end] = self.uni_val
        pad_len = L - res_len
        res_p = torch.cat([res_p, self._uniform_pad[:pad_len]], dim=0)

        sampled_tokens = torch.multinomial(res_p, self.tree_width).T
        if (~rej_vector).any():
            st_view = sampled_tokens[:, :res_len]
            st_view[1:] = torch.where(rej_vector, st_view[1:], st_view[:-1])
            st_view[0] = torch.where(
                rej_vector, st_view[0], d_tkns[target_rows, target_cols]
            )

        if res_tree_len > 0:
            res_rows[:res_tree_len] = self.first_siblings[
                res_rows[:res_tree_len], target_cols[:res_tree_len]
            ]
        res_p[:res_len] = p[res_rows, target_cols]

        return acc_len + 1, acc_row, sampled_tokens, res_p[None]


def push_forward_model_kwargs_and_inputs(
    model_inputs,
    collected_input_ids,
    model_input_ids,
    tree_mask,
    tree_pos_ids,
    acc_len,
    acc_row,
    additional_tokens,
    retrieve_indices,
):
    additional_tokens = additional_tokens[None]
    verified_input_ids = torch.cat(
        [
            collected_input_ids,
            model_input_ids[None, acc_row, 1:acc_len],
            additional_tokens[:, :1],
        ],
        dim=-1,
    )

    attn_mask = model_inputs["attention_mask"]
    position_ids = model_inputs["position_ids"]
    past_key_values = model_inputs["past_key_values"]

    bs, old_partial_len, old_full_len = attn_mask.shape
    device = attn_mask.device
    useful_full_len = verified_input_ids.shape[-1] - 1
    trash_len = old_full_len - useful_full_len

    tree_len = additional_tokens.shape[-1]
    new_full_len = useful_full_len + tree_len

    if trash_len > 0:
        attn_mask = attn_mask[:, :-trash_len, :-trash_len]
        position_ids = position_ids[:, :-trash_len]

    new_mask = torch.ones((bs, tree_len, new_full_len), dtype=bool, device=device)
    new_mask[..., :-tree_len] = attn_mask[:, -1:, :]
    new_mask[..., -tree_len:] = tree_mask

    new_position_ids = position_ids[:, -1:] + 1 + tree_pos_ids

    if acc_row != 0 and trash_len > 0:
        idx_tensor = -old_partial_len + retrieve_indices[acc_row, :acc_len]
        for l in range(len(past_key_values.key_cache)):
            past_key_values.key_cache[l][..., -old_partial_len:-trash_len, :] = (
                past_key_values.key_cache[l][..., idx_tensor, :]
            )
            past_key_values.value_cache[l][..., -old_partial_len:-trash_len, :] = (
                past_key_values.value_cache[l][..., idx_tensor, :]
            )
    if trash_len > 0:
        delete_false_key_value(past_key_values, trash_len)

    model_inputs = {
        "input_ids": additional_tokens,
        "attention_mask": new_mask,
        "position_ids": new_position_ids,
        "past_key_values": past_key_values,
        "cache_position": torch.arange(useful_full_len, new_full_len, device=device),
        "use_cache": model_inputs["use_cache"],
        "output_attentions": model_inputs["output_attentions"],
        "output_hidden_states": model_inputs["output_hidden_states"],
    }
    return model_inputs, verified_input_ids


def renew_pipeline(model_class):
    class SJDPACPipeline(model_class):

        def _init_new_params(
            self, guidance_scale=3.0, image_top_k=2000, text_top_k=10, **kwargs
        ):
            self.cfg = guidance_scale
            self.image_top_k = image_top_k
            self.text_top_k = text_top_k

        def create_logits_processor(self, cfg=3.0, image_top_k=2000, text_top_k=10):
            cfg = self.cfg if hasattr(self, "cfg") else cfg
            image_top_k = (
                self.image_top_k if hasattr(self, "image_top_k") else image_top_k
            )
            text_top_k = self.text_top_k if hasattr(self, "text_top_k") else text_top_k

            logits_processor = LogitsProcessorList()

            candidate_processor = MultiTokensVLLogitsProcessor(
                image_start_token_id=self.item_processor.token2id(
                    self.item_processor.image_start_token
                ),
                image_end_token_id=self.item_processor.token2id(
                    self.item_processor.image_end_token
                ),
                image_next_line_token_id=self.item_processor.token2id(
                    self.item_processor.new_line_token
                ),
                patch_size=32,
                voc_size=self.model.config.vocab_size,
                device=self.device,
            )

            topk_processor = MultiTokensInterleavedTopKLogitsWarper(
                image_top_k=image_top_k,
                text_top_k=text_top_k,
                image_start_token_id=self.item_processor.token2id(
                    self.item_processor.image_start_token
                ),
                image_end_token_id=self.item_processor.token2id(
                    self.item_processor.image_end_token
                ),
            )

            logits_processor.append(candidate_processor)
            logits_processor.append(topk_processor)

            return logits_processor

    return SJDPACPipeline


def get_multi_token_for_preparation(
    img_vocab, vocab_size, rand_token_num, input_ids, device
):
    img_vocab = img_vocab.to(device)
    img_vocab_size = len(img_vocab)
    rand_tokens = torch.randint(
        0, img_vocab_size, (*input_ids.shape[:-1], rand_token_num), device=device
    )
    rand_tokens = img_vocab[rand_tokens]

    scores_of_rand_tokens = torch.zeros((*rand_tokens.shape, vocab_size), device=device)
    scores_of_rand_tokens[..., img_vocab] = 1.0 / img_vocab_size

    return rand_tokens, scores_of_rand_tokens


def generate_tree_mask_and_retrieve(L: int, tree_width: int, tree_depth: int):
    if tree_width > 1:
        tree_nodes = (tree_width**tree_depth - 1) // (tree_width - 1)
    else:
        tree_nodes = tree_depth

    chain_nodes = L - tree_nodes
    assert (
        chain_nodes >= 0
    ), f"Tree too large for sequence length: {tree_nodes} nodes needed, but only {L} available."

    M = tree_depth + chain_nodes

    parents = [-1] * L
    for i in range(1, M):
        parents[i] = i - 1

    tree_indices = torch.zeros(L, dtype=torch.long)
    for i in range(M):
        tree_indices[i] = i

    if tree_width > 1:
        next_idx = M
        queue = []
        for i in range(tree_depth - 2, -1, -1):
            for b in range(1, tree_width):
                parents[next_idx] = i
                queue.append((next_idx, i + 1))
                tree_indices[next_idx] = b * M + (i + 1)
                next_idx += 1

        while queue:
            curr, curr_depth = queue.pop(0)
            if curr_depth < tree_depth - 1:
                for b in range(tree_width):
                    parents[next_idx] = curr
                    queue.append((next_idx, curr_depth + 1))
                    tree_indices[next_idx] = b * M + (curr_depth + 1)
                    next_idx += 1

    attention_mask = torch.zeros((L, L), dtype=torch.bool)
    for i in range(L):
        curr = i
        while curr != -1:
            attention_mask[i, curr] = True
            curr = parents[curr]

    is_parent = set(parents)
    leaves = [i for i in range(L) if i not in is_parent]
    retrieve_indices_list = []
    for leaf in leaves:
        path = []
        curr = leaf
        while curr != -1:
            path.append(curr)
            curr = parents[curr]
        path.reverse()
        retrieve_indices_list.append(path)

    num_leaves = tree_width ** (tree_depth - 1)
    assert len(retrieve_indices_list) == num_leaves

    max_len = max(len(p) for p in retrieve_indices_list) if retrieve_indices_list else 0
    retrieve_indices = torch.full((num_leaves, max_len), -1, dtype=torch.long)
    for i, path in enumerate(retrieve_indices_list):
        retrieve_indices[i, : len(path)] = torch.tensor(path, dtype=torch.long)

    return attention_mask, retrieve_indices, tree_indices


def setup_speculative_tree_buffers(
    prefix_token_sampler, tree_width, tree_depth, device
):
    """Precompute the tree index buffers used by :class:`SJDPACSpeculativeSampler`.

    This is only meaningful when ``tree_width >= 2`` (i.e. there is actual
    branching to speculate over). For the plain auto-regressive baseline the
    sampler is bypassed entirely, so this setup is skipped.
    """
    num_leaves = tree_width ** (tree_depth - 1)
    num_layers = tree_depth - 1
    leaf_indices = torch.arange(num_leaves, device=device)
    layer_indices = torch.arange(num_layers, device=device)
    layer_strides = tree_width ** (num_layers - 1 - layer_indices)
    branch_indices = (
        leaf_indices.view(-1, 1) // layer_strides.view(1, -1)
    ) % tree_width
    relative_offsets = torch.arange(tree_width - 1, device=device)
    offset_steps = relative_offsets + 1
    valid_ancestor_mask = relative_offsets.view(1, 1, -1) < branch_indices.unsqueeze(-1)
    ancestor_nodes = leaf_indices.view(-1, 1, 1) - offset_steps.view(
        1, 1, -1
    ) * layer_strides.view(1, -1, 1)
    valid_leaf_idx, valid_layer_idx, valid_offset = torch.where(valid_ancestor_mask)
    valid_ancestor_idx = ancestor_nodes[valid_ancestor_mask]
    direct_parent_mask = valid_ancestor_mask[:, :, 0]
    direct_leaf_idx, direct_layer_idx = torch.where(direct_parent_mask)
    direct_parent_idx = ancestor_nodes[:, :, 0][direct_parent_mask]
    block_sizes = layer_strides * tree_width
    first_siblings = (
        leaf_indices.view(-1, 1) // block_sizes.view(1, -1)
    ) * block_sizes.view(1, -1)
    last_siblings = first_siblings + block_sizes.view(1, -1) - 1
    node_group_heads = (
        leaf_indices.view(-1, 1) // layer_strides.view(1, -1)
    ) * layer_strides.view(1, -1)
    seq_curr_b, seq_curr_c, seq_prev_b = [], [], []
    for k in range(1, tree_width):
        mask = branch_indices == k
        b_idx, c_idx = torch.where(mask)
        seq_curr_b.append(b_idx)
        seq_curr_c.append(c_idx)
        seq_prev_b.append(b_idx - layer_strides[c_idx])

    prefix_token_sampler.tree_width = tree_width
    prefix_token_sampler.tree_depth = tree_depth
    prefix_token_sampler.b_idx = valid_leaf_idx
    prefix_token_sampler.c_idx = valid_layer_idx
    prefix_token_sampler.p_idx = valid_ancestor_idx
    prefix_token_sampler.u_b_idx = direct_leaf_idx
    prefix_token_sampler.u_c_idx = direct_layer_idx
    prefix_token_sampler.main_p_idx = direct_parent_idx
    prefix_token_sampler.first_siblings = first_siblings
    prefix_token_sampler.last_siblings = last_siblings
    prefix_token_sampler.node_group_heads = node_group_heads
    prefix_token_sampler.seq_curr_b = seq_curr_b
    prefix_token_sampler.seq_curr_c = seq_curr_c
    prefix_token_sampler.seq_prev_b = seq_prev_b


def renew_sampler(model_class):

    class SJDPACSampler(model_class, nn.Module):

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._init_new_params()

        def prepare_cfg_input(
            self,
            model_inputs,
            cfg_repeat_name_list,
            prefill_num=None,
            neg_input_ids=None,
        ):
            def cfg_repeat(x):
                return x.repeat(2, *([1] * (len(x.shape) - 1)))

            for name in cfg_repeat_name_list:
                if (name in model_inputs) and (model_inputs[name] is not None):

                    if name == "attention_mask":
                        model_inputs[name] = cfg_repeat(model_inputs[name])
                        B = model_inputs[name].shape[0]
                        model_inputs[name][B // 2 :, :prefill_num] = 0
                    elif name == "input_ids" and neg_input_ids is not None:
                        input_ids = model_inputs[name]
                        neg_input_ids = neg_input_ids
                        model_inputs[name] = get_double_cfg_input_ids(
                            input_ids,
                            neg_input_ids,
                            pad_category=self.config.pad_token_id,
                        )
                    else:
                        model_inputs[name] = cfg_repeat(model_inputs[name])

            return model_inputs

        def _get_initial_cache_position(self, input_ids, model_kwargs):
            """Calculates `cache_position` for the pre-fill stage based on `input_ids` and optionally past length"""
            # `torch.compile`-friendly `torch.arange` from a shape -- the lines below are equivalent to `torch.arange`
            if "inputs_embeds" in model_kwargs:
                cache_position = (
                    torch.ones_like(
                        model_kwargs["inputs_embeds"][0, :, 0], dtype=torch.int64
                    ).cumsum(0)
                    - 1
                )
            else:
                cache_position = (
                    torch.ones_like(input_ids[0, :], dtype=torch.int64).cumsum(0) - 1
                )

            if model_kwargs.get("past_key_values") is not None:
                cache = model_kwargs["past_key_values"]
                past_length = 0
                if not isinstance(cache, Cache):
                    past_length = cache[0][0].shape[2]
                elif (
                    hasattr(cache, "get_seq_length")
                    and cache.get_seq_length() is not None
                ):
                    past_length = cache.get_seq_length()

                if not is_torchdynamo_compiling():
                    cache_position = cache_position[past_length:]

            model_kwargs["cache_position"] = cache_position

            return model_kwargs

        def _init_new_params(
            self,
            jacobi_loop_interval_l=1,
            jacobi_loop_interval_r=(768 // 16) ** 2
            + 768 // 16,  # This should be determined by the image size ###!!!
            max_num_new_tokens=64,
            tree_width=3,
            tree_depth=3,
            guidance_scale=3.0,
            seed=42,
            do_cfg=True,
            use_chameleon_tokenizer=True,
            _init_doubled_attn_mask_cfg=False,
            **kwargs,
        ):
            if use_chameleon_tokenizer:
                import model.chameleon_vae_ori as chameleon_vae_ori

                chameleon_ori_vocab = chameleon_vae_ori.VocabInfo(
                    json.load(open("./ckpts/chameleon/tokenizer/text_tokenizer.json"))[
                        "model"
                    ]["vocab"]
                )
                chameleon_ori_translation = chameleon_vae_ori.VocabTranslation(
                    chameleon_ori_vocab
                )
                img_vocab = chameleon_ori_translation._vocab.image_tokens
                self.register_buffer(
                    "img_vocab", torch.tensor(img_vocab, dtype=torch.long)
                )
            else:
                if not hasattr(self, "img_vocab"):
                    self.img_vocab = None

            self.cfg_repeat_name_list = [
                "inputs_embeds",
                "input_ids",
                "pixel_values",
            ]
            self.cfg_half_name_list = [
                "inputs_embeds",
                "input_ids",
                "pixel_values",
            ]
            self.jacobi_loop_interval_l = jacobi_loop_interval_l
            self.jacobi_loop_interval_r = jacobi_loop_interval_r
            self.max_num_new_tokens = max_num_new_tokens
            self.max_jacobi_iter_num = min(200, self.max_num_new_tokens + 1)
            self.tree_width = tree_width
            self.tree_depth = tree_depth
            self.guidance_scale = guidance_scale

            self.seed = seed
            self.generator = None
            self.do_cfg = do_cfg
            self._init_doubled_attn_mask_cfg = _init_doubled_attn_mask_cfg

        def _sample(
            self,
            input_ids: torch.LongTensor,
            logits_processor: LogitsProcessorList,
            stopping_criteria: StoppingCriteriaList,
            generation_config: GenerationConfig,
            synced_gpus: bool,
            streamer,
            logits_warper: Optional[LogitsProcessorList] = None,
            **model_kwargs,
        ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
            r"""
            Generates sequences of token ids for models with a language modeling head using **multinomial sampling** and
            can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

            Parameters:
                input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                    The sequence used as a prompt for the generation.
                logits_processor (`LogitsProcessorList`):
                    An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                    used to modify the prediction scores of the language modeling head applied at each generation step.
                stopping_criteria (`StoppingCriteriaList`):
                    An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                    used to tell if the generation loop should stop.
                generation_config ([`~generation.GenerationConfig`]):
                    The generation configuration to be used as parametrization of the decoding method.
                synced_gpus (`bool`):
                    Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
                streamer (`BaseStreamer`, *optional*):
                    Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                    through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
                logits_warper (`LogitsProcessorList`, *optional*):
                    An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsWarper`] used
                    to warp the prediction score distribution of the language modeling head applied before multinomial
                    sampling at each generation step. Only required with sampling strategies (i.e. `do_sample` is set in
                    `generation_config`)
                model_kwargs:
                    Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
                    an encoder-decoder model the kwargs should include `encoder_outputs`.

            Return:
                [`~generation.GenerateDecoderOnlyOutput`], [`~generation.GenerateEncoderDecoderOutput`] or `torch.LongTensor`:
                A `torch.LongTensor` containing the generated tokens (default behaviour) or a
                [`~generation.GenerateDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
                `return_dict_in_generate=True` or a [`~generation.GenerateEncoderDecoderOutput`] if
                `model.config.is_encoder_decoder=True`.
            """
            # init values
            pad_token_id = generation_config._pad_token_tensor
            output_attentions = generation_config.output_attentions
            output_hidden_states = generation_config.output_hidden_states
            output_scores = generation_config.output_scores
            output_logits = generation_config.output_logits
            return_dict_in_generate = generation_config.return_dict_in_generate
            max_length = generation_config.max_length

            for c in stopping_criteria:
                if isinstance(c, EosTokenCriteria):
                    c.eos_token_id[0] = logits_processor[0].image_end_token_id
                    c.__class__ = SpecEosCriteria

            # init attention / hidden states / scores tuples
            scores = () if (return_dict_in_generate and output_scores) else None
            raw_logits = () if (return_dict_in_generate and output_logits) else None
            decoder_attentions = (
                () if (return_dict_in_generate and output_attentions) else None
            )
            cross_attentions = (
                () if (return_dict_in_generate and output_attentions) else None
            )
            decoder_hidden_states = (
                () if (return_dict_in_generate and output_hidden_states) else None
            )

            # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
            if return_dict_in_generate and self.config.is_encoder_decoder:
                encoder_attentions = (
                    model_kwargs["encoder_outputs"].get("attentions")
                    if output_attentions
                    else None
                )
                encoder_hidden_states = (
                    model_kwargs["encoder_outputs"].get("hidden_states")
                    if output_hidden_states
                    else None
                )

            device = input_ids.device
            dtype = input_ids.dtype

            # keep track of which sequences are already finished
            batch_size, cur_len = input_ids.shape
            this_peer_finished = False
            unfinished_sequences = torch.ones(batch_size, dtype=dtype, device=device)

            # init: attn mask, cache_position, cfg,
            model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)
            prefill_num = model_kwargs["attention_mask"].shape[1] - 1

            do_cfg = self.do_cfg if hasattr(self, "do_cfg") else False

            guidance_scale = (
                self.guidance_scale if hasattr(self, "guidance_scale") else 3.0
            )
            do_cfg = do_cfg & (guidance_scale != 1)

            if do_cfg:
                model_kwargs = self.prepare_cfg_input(
                    model_kwargs,
                    cfg_repeat_name_list=(
                        [
                            "attention_mask",
                        ]
                        if (not self._init_doubled_attn_mask_cfg)
                        else []
                    ),
                    prefill_num=prefill_num,
                )

            if self.seed is not None:
                set_seed(self.seed)
                self.generator = torch.Generator(device).manual_seed(self.seed)

            gen_loop_num = 0

            prefix_token_sampler = SJDPACSpeculativeSampler(generator=self.generator)

            tree_width = self.tree_width
            tree_depth = self.tree_depth

            # ``max_num_new_tokens == 1`` selects the plain auto-regressive
            # baseline: one token per forward pass, no speculation. The tree
            # collapses to a single node and the speculative sampler is bypassed.
            is_baseline = self.max_num_new_tokens <= 1
            if is_baseline:
                tree_width = 1
                tree_depth = 1
            else:
                setup_speculative_tree_buffers(
                    prefix_token_sampler, tree_width, tree_depth, device
                )

            additional_tokens, additional_scores = get_multi_token_for_preparation(
                self.img_vocab,
                self.config.vocab_size,
                self.max_num_new_tokens - 1,
                input_ids,
                device,
            )
            tree_mask, retrieve_indices, from_tree_ids = (
                generate_tree_mask_and_retrieve(
                    self.max_num_new_tokens, tree_width, tree_depth
                )
            )
            tree_mask = tree_mask.to(device)
            retrieve_indices = retrieve_indices.to(device)
            from_tree_ids = from_tree_ids.to(device)
            tree_pos_ids = tree_mask.sum(-1) - 1

            new_ids = torch.cat(
                [input_ids, additional_tokens.expand(input_ids.size(0), -1)], dim=1
            )
            attn_mask = model_kwargs["attention_mask"]
            new_mask = torch.zeros(
                (attn_mask.shape[0], new_ids.shape[1], new_ids.shape[1]),
                device=device,
                dtype=torch.bool,
            )
            new_mask[:, : attn_mask.shape[1], : attn_mask.shape[1]] = torch.tril(
                attn_mask.unsqueeze(-2) & attn_mask.unsqueeze(-1)
            )
            new_mask[:, attn_mask.shape[1] :, : attn_mask.shape[1]] = new_mask[
                :, attn_mask.shape[1] - 1 : attn_mask.shape[1], : attn_mask.shape[1]
            ]
            new_mask[:, attn_mask.shape[1] :, attn_mask.shape[1] :] = (
                tree_mask[1:, 1:].unsqueeze(0).expand(attn_mask.shape[0], -1, -1)
            )
            cache_position = torch.arange(new_ids.shape[1], device=device)
            input_token_scores = torch.hstack(
                (torch.zeros_like(additional_scores[:, :1]), additional_scores)
            )
            model_inputs = {
                "input_ids": new_ids.contiguous(),
                "attention_mask": new_mask,
                "position_ids": new_mask.sum(-1).clamp_min(1) - 1,
                "past_key_values": model_kwargs["past_key_values"],
                "cache_position": cache_position,
                "use_cache": model_kwargs["use_cache"],
                "output_attentions": output_attentions,
                "output_hidden_states": output_hidden_states,
            }

            count_time = True
            if count_time:
                t1 = torch.cuda.Event(enable_timing=True)
                t2 = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                t1.record()

            while self._has_unfinished_sequences(
                this_peer_finished,
                synced_gpus,
                device=device,
                cur_len=cur_len,
                max_length=max_length,
            ):
                # start = time.time()
                # idx = 1

                # the first element of model_inputs['input_ids'] is in all_collected_input_ids
                all_collected_input_ids = input_ids
                model_input_ids = model_inputs["input_ids"]

                is_force_no_cfg = check_is_force_no_cfg(
                    input_ids,
                    image_start_token_id=(
                        logits_processor[0].image_start_token_id
                        if hasattr(logits_processor[0], "image_start_token_id")
                        else None
                    ),
                    image_end_token_id=(
                        logits_processor[0].image_end_token_id
                        if hasattr(logits_processor[0], "image_end_token_id")
                        else None
                    ),
                )
                if do_cfg:
                    model_inputs = self.prepare_cfg_input(
                        model_inputs,
                        cfg_repeat_name_list=self.cfg_repeat_name_list,
                        neg_input_ids=(
                            model_kwargs.get("neg_input_ids", None)
                            if (gen_loop_num == 0)
                            else None
                        ),
                    )

                # print(f"{idx}: {time.time()-start}")
                # start = time.time()
                # idx += 1

                # forward pass to get next token
                outputs = self(**model_inputs, return_dict=True)

                # print(f"{idx}: {time.time()-start}")
                # start = time.time()
                # idx += 1

                if synced_gpus and this_peer_finished:
                    continue  # don't waste resources running the code we don't need

                logits = outputs.logits[:, -self.max_num_new_tokens :]
                conditional_logits, unconditional_logits = logits.chunk(2, dim=0)
                conditional_logits = conditional_logits[0, retrieve_indices]
                unconditional_logits = unconditional_logits[0, retrieve_indices]

                model_input_ids = model_input_ids[:, -self.max_num_new_tokens :]
                model_input_ids = torch.hstack(
                    (model_input_ids, -torch.ones_like(model_input_ids[:, -1:]))
                )
                model_input_ids = model_input_ids[0, retrieve_indices]

                if do_cfg:
                    if is_force_no_cfg:
                        next_token_logits = conditional_logits
                    else:
                        next_token_logits = (
                            guidance_scale * (conditional_logits - unconditional_logits)
                            + unconditional_logits
                        )

                next_token_logits = logits_processor(
                    all_collected_input_ids, next_token_logits
                )
                if logits_warper is not None:
                    next_token_logits = logits_warper(
                        all_collected_input_ids, next_token_logits
                    )
                next_token_scores = next_token_logits.softmax(dim=-1)

                # print(f"{idx}: {time.time()-start}")
                # start = time.time()
                # idx += 1

                if do_cfg:
                    model_inputs = postprocess_cfg_decode(model_inputs)

                if is_baseline:
                    # Standard AR: sample the single next token and accept it.
                    next_token = torch.multinomial(
                        next_token_scores[:, -1], 1, generator=self.generator
                    )
                    acc_len, acc_row = 1, 0
                    additional_tokens = next_token
                    additional_scores = next_token_scores
                else:
                    acc_len, acc_row, additional_tokens, additional_scores = (
                        prefix_token_sampler(
                            draft_tokens=model_input_ids,
                            draft_prob=input_token_scores,
                            advanced_prob=next_token_scores,
                        )
                    )
                additional_tokens = additional_tokens.flatten()[from_tree_ids]
                input_token_scores = additional_scores

                (
                    model_inputs,
                    updated_input_ids,
                ) = push_forward_model_kwargs_and_inputs(
                    model_inputs=model_inputs,
                    collected_input_ids=all_collected_input_ids,
                    model_input_ids=model_input_ids,
                    tree_mask=tree_mask,
                    tree_pos_ids=tree_pos_ids,
                    acc_len=acc_len,
                    acc_row=acc_row,
                    additional_tokens=additional_tokens,
                    retrieve_indices=retrieve_indices,
                )
                input_ids = updated_input_ids

                # print(f"{idx}: {time.time()-start}")
                # start = time.time()
                # idx += 1

                assert not return_dict_in_generate

                # check whether we get the end token
                unfinished_sequences = unfinished_sequences & ~stopping_criteria(
                    input_ids, scores
                )
                this_peer_finished = unfinished_sequences.max() == 0

                cur_len = input_ids.shape[1]
                gen_loop_num += 1

                # This is needed to properly delete outputs.logits which may be very large for first iteration
                # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
                del outputs

                # print(f"{idx}: {time.time()-start}")
                # start = time.time()
                # idx += 1

            if streamer is not None:
                streamer.end()

            if count_time:
                t2.record()
                torch.cuda.synchronize()

                t = t1.elapsed_time(t2) / 1000
                print("Time elapsed inner: ", t)
                print("gen loop num (NFE): ", gen_loop_num)
                print("tokens length: ", cur_len)
                logging.info(f"Time elapsed inner: {t}")
                logging.info(f"gen loop num (NFE): {gen_loop_num}")
                logging.info(f"tokens length: {cur_len}")

            if return_dict_in_generate:
                if self.config.is_encoder_decoder:
                    return GenerateEncoderDecoderOutput(
                        sequences=input_ids,
                        scores=scores,
                        logits=raw_logits,
                        encoder_attentions=encoder_attentions,
                        encoder_hidden_states=encoder_hidden_states,
                        decoder_attentions=decoder_attentions,
                        cross_attentions=cross_attentions,
                        decoder_hidden_states=decoder_hidden_states,
                        past_key_values=model_kwargs.get("past_key_values"),
                    )
                else:
                    return GenerateDecoderOnlyOutput(
                        sequences=input_ids,
                        scores=scores,
                        logits=raw_logits,
                        attentions=decoder_attentions,
                        hidden_states=decoder_hidden_states,
                        past_key_values=model_kwargs.get("past_key_values"),
                    )
            else:
                return input_ids

    return SJDPACSampler


def renew_backbone(model_class):
    class SJDPACBackbone(model_class):

        def _update_causal_mask(
            self,
            attention_mask: torch.Tensor,
            input_tensor: torch.Tensor,
            cache_position: torch.Tensor,
            past_key_values: Cache,
            output_attentions: bool,
        ):
            # TODO: As of torch==2.2.0, the `attention_mask` passed to the model in `generate` is 2D and of dynamic length even when the static
            # KV cache is used. This is an issue for torch.compile which then recaptures cudagraphs at each decode steps due to the dynamic shapes.
            # (`recording cudagraph tree for symint key 13`, etc.), which is VERY slow. A workaround is `@torch.compiler.disable`, but this prevents using
            # `fullgraph=True`. See more context in https://github.com/huggingface/transformers/pull/29114

            if self.config._attn_implementation == "flash_attention_2":
                if attention_mask is not None and 0.0 in attention_mask:
                    return attention_mask
                return None

            # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
            # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
            # to infer the attention mask.
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            using_static_cache = isinstance(past_key_values, StaticCache)

            # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
            if (
                self.config._attn_implementation == "sdpa"
                and not using_static_cache
                and not output_attentions
            ):
                if AttentionMaskConverter._ignore_causal_mask_sdpa(
                    attention_mask,
                    inputs_embeds=input_tensor,
                    past_key_values_length=past_seen_tokens,
                    is_training=self.training,
                ):
                    return None

            dtype, device = input_tensor.dtype, input_tensor.device
            min_dtype = torch.finfo(dtype).min
            sequence_length = input_tensor.shape[1]
            if using_static_cache:
                target_length = past_key_values.get_max_length()
            else:
                target_length = (
                    attention_mask.shape[-1]
                    if isinstance(attention_mask, torch.Tensor)
                    else past_seen_tokens + sequence_length + 1
                )

            if attention_mask is not None and attention_mask.dim() == 4:
                # in this case we assume that the mask comes already in inverted form and requires no inversion or slicing
                if attention_mask.max() != 0:
                    raise ValueError(
                        "Custom 4D attention mask should be passed in inverted form with max==0`"
                    )
                causal_mask = attention_mask
            else:
                causal_mask = torch.full(
                    (sequence_length, target_length),
                    fill_value=min_dtype,
                    dtype=dtype,
                    device=device,
                )
                if sequence_length != 1:
                    causal_mask = torch.triu(causal_mask, diagonal=1)
                causal_mask *= torch.arange(
                    target_length, device=device
                ) > cache_position.reshape(-1, 1)
                causal_mask = causal_mask[None, None, :, :].expand(
                    input_tensor.shape[0], 1, -1, -1
                )
                if attention_mask is not None:
                    causal_mask = (
                        causal_mask.clone()
                    )  # copy to contiguous memory for in-place edit
                    mask_length = attention_mask.shape[-1]

                    while attention_mask.dim() < 4:
                        attention_mask = attention_mask.unsqueeze(1)

                    padding_mask = (
                        causal_mask[:, :, :, :mask_length] + attention_mask
                    )  # [:, None, None, :]
                    padding_mask = padding_mask == 0
                    causal_mask[:, :, :, :mask_length] = causal_mask[
                        :, :, :, :mask_length
                    ].masked_fill(padding_mask, min_dtype)
            if (
                self.config._attn_implementation == "sdpa"
                and attention_mask is not None
                and attention_mask.device.type == "cuda"
                and not output_attentions
            ):
                # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
                # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
                # Details: https://github.com/pytorch/pytorch/issues/110213
                causal_mask = AttentionMaskConverter._unmask_unattended(
                    causal_mask, min_dtype
                )

            return causal_mask

    return SJDPACBackbone


def renew_pipeline_sampler(pipe_line, **kwargs):
    pipe_line.__class__ = renew_pipeline(pipe_line.__class__)
    pipe_line._init_new_params(**kwargs)
    pipe_line.model.__class__ = renew_sampler(pipe_line.model.__class__)
    pipe_line.model._init_new_params(**kwargs)
    pipe_line.model.model.__class__ = renew_backbone(pipe_line.model.model.__class__)
    return pipe_line
