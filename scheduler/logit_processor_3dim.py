import math

import torch
from transformers.generation.logits_process import (
    LogitsProcessor,
    LogitsWarper,
)


def check_eol_in_multitokens(tokenlen, new_pred_tokenlen, line_len):
    L, R = (tokenlen + 1), (tokenlen + new_pred_tokenlen)
    check_interval_l = L // line_len + 1 if L % line_len != 0 else L // line_len
    check_interval_r = R // line_len
    return check_interval_l <= check_interval_r


def get_eol_in_multitokens(
    logits, eol_cls, tokenlen, new_pred_tokenlen, line_len, min_dtype=-math.inf
):
    logits_forced_eol = logits.clone()
    L, R = (tokenlen + 1), (tokenlen + new_pred_tokenlen)
    check_interval_l = L // line_len + 1 if L % line_len != 0 else L // line_len
    check_interval_r = R // line_len
    eol_position_ids = [
        line_len * multi_num - (tokenlen + 1)
        for multi_num in range(check_interval_l, check_interval_r + 1)
    ]
    for i in eol_position_ids:
        logits_forced_eol[..., i, :] = min_dtype
        logits_forced_eol[..., i, eol_cls] = 0

    return logits_forced_eol, eol_position_ids


class MultiTokensVLLogitsProcessor(LogitsProcessor):

    def __init__(
        self,
        image_start_token_id=None,
        image_end_token_id=None,
        image_next_line_token_id=None,
        patch_size=None,
        voc_size=None,
        device="cpu",
    ):
        self.image_start_token_id = image_start_token_id  # 8197
        self.image_end_token_id = image_end_token_id  # 8196
        self.image_next_line_token_id = image_next_line_token_id  # 8803
        self.image_start_token_id_index = None
        self.patch_size = patch_size
        self.h_latent_dim = None
        self.w_latent_dim = None

        self.vocab_list = [i for i in range(voc_size)]
        self.image_token_list = [i for i in range(4, 8195 + 1)]
        self.suppress_tokens = torch.tensor(
            [x for x in self.vocab_list if x not in self.image_token_list],
            device=device,
        )

        self.vocab_tensor = torch.arange(voc_size, device=device)
        self.suppress_token_mask = torch.isin(
            self.vocab_tensor, self.suppress_tokens
        )  # not [   4,    5,    6,  ..., 8193, 8194, 8195]
        self.new_line_force_token_mask = torch.isin(
            self.vocab_tensor,
            torch.tensor([self.image_next_line_token_id], device=device),
        )
        self.eos_image_force_token_mask = torch.isin(
            self.vocab_tensor, torch.tensor([self.image_end_token_id], device=device)
        )

        self.flag = False
        self.num_image_start_tokens = None
        self.num_image_end_tokens = None

    # @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:

        self.num_image_start_tokens = (input_ids[0] == self.image_start_token_id).sum()
        self.num_image_end_tokens = (input_ids[0] == self.image_end_token_id).sum()

        if self.num_image_start_tokens == self.num_image_end_tokens:
            self.h_latent_dim, self.w_latent_dim = None, None
            self.image_start_token_id_index = None
            return scores

        elif self.num_image_start_tokens == self.num_image_end_tokens + 1:
            if self.image_start_token_id_index is None:
                # self.image_start_token_id_index = torch.where(
                #     input_ids[0] == self.image_start_token_id
                # )[0]
                self.image_start_token_id_index = torch.where(
                    input_ids[0] == self.image_start_token_id
                )[0][-1].item()

            new_logit_token_len = scores.shape[-2] if scores.ndim >= 3 else 1

            new_token_num = len(input_ids[0][self.image_start_token_id_index + 1 :])
            if new_token_num >= 2:

                pad_eol_len = 1  # TODO: to check

                if self.h_latent_dim is None or self.w_latent_dim is None:
                    h_grids, w_grids = (
                        input_ids[0][self.image_start_token_id_index + 1] - 8804,
                        input_ids[0][self.image_start_token_id_index + 2] - 8804,
                    )
                    self.h_latent_dim, self.w_latent_dim = h_grids * 2, w_grids * 2
                    print(
                        "self.h_latent_dim, self.w_latent_dim",
                        self.h_latent_dim,
                        self.w_latent_dim,
                    )

                tokens = input_ids[0][self.image_start_token_id_index + 3 :]

                is_new_seq_ids_containing_end_of_line = check_eol_in_multitokens(
                    len(tokens), new_logit_token_len, self.w_latent_dim + pad_eol_len
                )
                is_new_seq_ids_containing_end_of_img = check_eol_in_multitokens(
                    len(tokens),
                    new_logit_token_len,
                    (self.w_latent_dim + pad_eol_len) * self.h_latent_dim + pad_eol_len,
                )

                # TODO: is_pre_seq_containing_end_of_img:
                scores = torch.where(
                    self.suppress_token_mask.to(scores.device), -float("inf"), scores
                )

                # containing ONE end-of-line
                if is_new_seq_ids_containing_end_of_line:

                    scores, eol_position_ids = get_eol_in_multitokens(
                        scores,
                        self.image_next_line_token_id,
                        len(tokens),
                        new_logit_token_len,
                        self.w_latent_dim + pad_eol_len,
                    )

                # containing ONE end-of-image
                if is_new_seq_ids_containing_end_of_img:
                    scores, eol_position_ids = get_eol_in_multitokens(
                        scores,
                        self.image_end_token_id,
                        len(tokens),
                        new_logit_token_len,
                        (self.w_latent_dim + pad_eol_len) * self.h_latent_dim
                        + pad_eol_len,
                    )

                return scores
        # else:
        #     print(
        #         f"Something wrong in the decoding process. MultiTokensVLLogitsProcessor. \
        #           st: id {torch.where(input_ids[0] == self.image_start_token_id)} num {self.num_image_start_tokens} \
        #           ed: id {torch.where(input_ids[0] == self.image_end_token_id)} num {self.num_image_end_tokens} \
        #           input_ids.shape {input_ids.shape} scores.shape {scores.shape} "
        #     )

        return scores


class MultiTokensInterleavedTopKLogitsWarper(LogitsWarper):
    r"""
    [`LogitsWarper`] that performs top-k, i.e. restricting to the k highest probability elements. Often used together
    with [`TemperatureLogitsWarper`] and [`TopPLogitsWarper`].
    """

    def __init__(
        self,
        image_top_k: int,
        text_top_k: int,
        image_start_token_id=None,
        image_end_token_id=None,
        filter_value: float = -float("Inf"),
        min_tokens_to_keep: int = 1,
    ):
        if not isinstance(text_top_k, int) or text_top_k <= 0:
            raise ValueError(
                f"`text_top_k` has to be a strictly positive integer, but is {text_top_k}"
            )
        if not isinstance(image_top_k, int) or text_top_k <= 0:
            raise ValueError(
                f"`image_top_k` has to be a strictly positive integer, but is {image_top_k}"
            )

        self.image_top_k = max(image_top_k, min_tokens_to_keep)
        self.text_top_k = max(text_top_k, min_tokens_to_keep)
        self.filter_value = filter_value

        self.image_start_token_id = image_start_token_id
        self.image_end_token_id = image_end_token_id

        self.flag = False
        self.num_image_start_tokens = None
        self.num_image_end_tokens = None

    # @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        num_starts = (input_ids == self.image_start_token_id).sum(dim=-1)
        num_ends = (input_ids == self.image_end_token_id).sum(dim=-1)
        is_in_image = (num_starts == num_ends + 1).unsqueeze(-1)

        img_threshold = torch.topk(scores, self.image_top_k)[0][..., -1, None]
        txt_threshold = torch.topk(scores, self.text_top_k)[0][..., -1, None]

        to_remove = scores < torch.where(is_in_image, img_threshold, txt_threshold)
        return scores.masked_fill(to_remove, self.filter_value)


def get_double_cfg_input_ids(input_ids, neg_input_ids, pad_category):
    batchsize, prefill_num = input_ids.shape

    neg_prefill_num = neg_input_ids.shape[1]

    batchsize_cfg = 2 * batchsize
    max_prefill_num = max(prefill_num, neg_prefill_num)

    new_neg_input_ids = torch.full(
        (batchsize_cfg, max_prefill_num),
        pad_category,
        dtype=input_ids.dtype,
        device=input_ids.device,
    )

    new_neg_input_ids[:batchsize, -input_ids.shape[1] :] = input_ids
    new_neg_input_ids[batchsize:, -neg_input_ids.shape[1] :] = neg_input_ids

    return new_neg_input_ids
