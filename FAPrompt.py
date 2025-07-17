import os
from typing import Union, List, Any
from pkg_resources import packaging
import torch
import numpy as np
from AnomalyCLIP_lib.simple_tokenizer import SimpleTokenizer as _Tokenizer
# from open_clip import tokenizer
# simple_tokenizer = tokenizer.SimpleTokenizer()
from copy import deepcopy
import torch.nn as nn
from collections import OrderedDict

_tokenizer = _Tokenizer()

def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False) -> Union[torch.IntTensor, torch.LongTensor]:
    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize

    context_length : int
        The context length to use; all CLIP models use 77 as the context length

    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length].
    We return LongTensor when torch version is <1.8.0, since older index_select requires indices to be long.
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    if packaging.version.parse(torch.__version__) < packaging.version.parse("1.8.0"):
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
    else:
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result

def _get_clones(module, N):
    return nn.ModuleList([deepcopy(module) for i in range(N)])

class FAPrompt(nn.Module):
    def __init__(self, clip_model, design_details):
        super().__init__()
        classnames = ["object"]
        self.n_cls = len(classnames)
        n_ctx_pos = 5
        n_ctx_neg = 2
        self.num_p = 10
        self.k = 10
        self.text_encoder_n_ctx = design_details["learnabel_text_embedding_length"]
        dtype = clip_model.transformer.get_cast_dtype()

        ctx_dim = clip_model.ln_final.weight.shape[0]
        self.classnames = classnames

        self.patch_meta_net = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(768 * self.k, 768 * self.k // 16)),  #768
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(768 * self.k // 16, 768))
        ]))

        # Random Initialization
        print("Initializing class-specific contexts")
        # 这里是cls是类的个数，n_ctx_pos代表learnable token的长度，ctx_dim表示prompt的dimension
        ctx_vectors_pos = torch.empty(self.n_cls, 1, n_ctx_pos, ctx_dim, dtype=dtype)
        ctx_vectors_neg = torch.empty(self.n_cls, self.num_p, n_ctx_neg, ctx_dim, dtype=dtype)
        nn.init.normal_(ctx_vectors_pos, std=0.02)
        nn.init.normal_(ctx_vectors_neg, std=0.02)
        prompt_prefix_pos = " ".join(["N"] * n_ctx_pos)
        prompt_prefix_neg = " ".join(["A"] * n_ctx_neg)
        self.compound_prompts_depth = design_details["learnabel_text_embedding_depth"]
        self.compound_prompts_text = nn.ParameterList([nn.Parameter(torch.empty(self.text_encoder_n_ctx, ctx_dim))
                                                       for _ in range(self.compound_prompts_depth - 1)])
        for single_para in self.compound_prompts_text:
            print("single_para", single_para.shape)
            nn.init.normal_(single_para, std=0.02)

        single_layer = nn.Linear(ctx_dim, 896)
        self.compound_prompt_projections = _get_clones(single_layer, self.compound_prompts_depth - 1)

        self.ctx_pos = nn.Parameter(ctx_vectors_pos)  # to be optimized
        self.ctx_neg = nn.Parameter(ctx_vectors_neg)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]

        prompts_pos = [prompt_prefix_pos + " " + name + "." for name in classnames]
        prompts_neg = [prompt_prefix_pos + " " + prompt_prefix_neg + " " + "damaged" + " " + name + "." for _ in
                       range(self.num_p) for name in classnames]

        tokenized_prompts_pos = []
        tokenized_prompts_neg = []

        for p_pos in prompts_pos:
            tokenized_prompts_pos.append(tokenize(p_pos))
        for p_neg in prompts_neg:
            tokenized_prompts_neg.append(tokenize(p_neg))
        tokenized_prompts_pos = torch.cat(tokenized_prompts_pos)
        tokenized_prompts_neg = torch.cat(tokenized_prompts_neg)
        # 生成相应的text embedding
        with torch.no_grad():
            embedding_pos = clip_model.token_embedding(tokenized_prompts_pos).type(dtype)
            embedding_neg = clip_model.token_embedding(tokenized_prompts_neg).type(dtype)
            n, l, d = embedding_pos.shape
            print("embedding_pos", embedding_pos.shape)
            embedding_pos = embedding_pos.reshape(1, self.n_cls, l, d).permute(1, 0, 2, 3)
            embedding_neg = embedding_neg.reshape(self.num_p, self.n_cls, l, d).permute(1, 0, 2, 3)

        self.register_buffer("token_prefix_pos", embedding_pos[:, :, :1, :])
        self.register_buffer("token_suffix_pos", embedding_pos[:, :, 1 + n_ctx_pos:, :])
        self.register_buffer("token_prefix_neg", embedding_neg[:, :, :1, :])
        self.register_buffer("token_suffix_neg", embedding_neg[:, :, 1 + n_ctx_pos + n_ctx_neg:, :])

        n, d = tokenized_prompts_pos.shape
        tokenized_prompts_pos = tokenized_prompts_pos.reshape(1, self.n_cls, d).permute(1, 0, 2)

        n, d = tokenized_prompts_neg.shape
        tokenized_prompts_neg = tokenized_prompts_neg.reshape(self.num_p, self.n_cls, d).permute(1, 0, 2)

        self.n_ctx_pos = n_ctx_pos
        self.n_ctx_neg = n_ctx_neg
        self.vis_dim = ctx_dim
        # tokenized_prompts = torch.cat([tokenized_prompts_pos, tokenized_prompts_neg], dim=0)  # torch.Tensor
        self.register_buffer("tokenized_prompts_pos", tokenized_prompts_pos)
        self.register_buffer("tokenized_prompts_neg", tokenized_prompts_neg)
        print("tokenized_prompts shape", self.tokenized_prompts_pos.shape, self.tokenized_prompts_neg.shape)

    def forward(self, selected_tokens = None):
        ctx_pos = self.ctx_pos
        # print("shape", ctx_pos.shape)
        prefix_pos = self.token_prefix_pos
        suffix_pos = self.token_suffix_pos

        prompts_pos = torch.cat(
            [
                # N(the number of template), 1, dim
                prefix_pos,  # (n_cls, 1, dim)
                ctx_pos,  # (n_cls, n_ctx, dim)
                suffix_pos,  # (n_cls, *, dim)
            ],
            dim=2,
        )

        ctx_neg = self.ctx_neg
        prefix_neg = self.token_prefix_neg
        suffix_neg = self.token_suffix_neg

        ctx_pos2 = ctx_pos.expand(-1, self.num_p, -1, -1).reshape(-1, self.num_p, self.n_ctx_pos, self.vis_dim)

        bias = 0
        if selected_tokens != None:
            selected_tokens = selected_tokens.reshape(1, self.vis_dim * self.k)
            bias = self.patch_meta_net(selected_tokens)
            ctx_neg = ctx_neg + bias

        prompts_neg = torch.cat(
            [
                prefix_neg,  # (n_cls, 1, dim)
                ctx_pos2,
                ctx_neg,  # (n_cls, n_ctx, dim)
                suffix_neg,  # (n_cls, *, dim)
            ],
            dim=2,
        )

        _, _, l, d = prompts_pos.shape
        prompts_pos = prompts_pos.reshape(-1, l, d)
        _, _, l, d = prompts_neg.shape
        prompts_neg = prompts_neg.reshape(-1, l, d)
        # prompts = torch.cat([prompts_pos, prompts_neg], dim=0)


        _, l, d = self.tokenized_prompts_pos.shape
        tokenized_prompts_pos = self.tokenized_prompts_pos.reshape(-1, d)
        _, l, d = self.tokenized_prompts_neg.shape
        tokenized_prompts_neg = self.tokenized_prompts_neg.reshape(-1, d)
        # tokenized_prompts = torch.cat((tokenized_prompts_pos, tokenized_prompts_neg), dim = 0)


        return prompts_pos, prompts_neg, tokenized_prompts_pos, tokenized_prompts_neg, self.compound_prompts_text, bias