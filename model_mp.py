from model import *
import os
import torch.distributed.rpc as rpc
from torch.distributed.rpc import RRef


class GPTPipelineBlock(nn.Module):
    def __init__(self, config, is_first, is_last, num_layers):
        super().__init__()
        self.config = config
        self.is_first = is_first
        self.is_last = is_last

        self.device = f"cuda:{os.environ['LOCAL_RANK']}"

        self.transformer = nn.ModuleDict()

        if self.is_first:
            self.transformer['wte'] = nn.Embedding(config.vocab_size, config.n_embd)
            self.transformer['wpe'] = nn.Embedding(config.block_size, config.n_embd)
            self.transformer['drop'] = nn.Dropout(config.dropout)

        self.transformer['h'] = nn.ModuleList([Block(config) for _ in range(num_layers)])

        if self.is_last:
            self.transformer['ln_f'] = LayerNorm(config.n_embd, bias=config.bias)
            self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        print(f"Initialized shard on {self.device} with {num_layers} layers (first={is_first}, last={is_last})")
        self.to(self.device)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def train(self, mode=True):
        super().train(mode)
        return self

    def eval(self):
        return self.train(False)

    def forward(self, x_rref):
        x = x_rref.to_here().to(self.device)

        if self.is_first:
            b, t = x.size()
            assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
            pos = torch.arange(0, t, dtype=torch.long, device=self.device)
            tok_emb = self.transformer.wte(x)
            pos_emb = self.transformer.wpe(pos)
            x = self.transformer.drop(tok_emb + pos_emb)

        for block in self.transformer.h:
            x = block(x)

        if self.is_last:
            x = self.transformer.ln_f(x)
            # Check for generate() mode
            if self.training is False and x.size(1) == 1:
                logits = self.lm_head(x[:, [-1], :])
            else:
                logits = self.lm_head(x)
            return logits.cpu()

        return x.cpu()

    def get_wte_weight(self):
        if self.is_first:
            return RRef(self.transformer.wte.weight)
        return None

    def tie_weights(self, wte_weight_rref):
        if self.is_last:
            self.lm_head.weight = wte_weight_rref.to_here().to(self.device)

    def parameter_rrefs(self):
        return [RRef(p) for p in self.parameters()]


class GPT_mp(GPT):
    """
    Coordinates the pipeline of remote shards from the master node.
    """
    def __init__(self, config, model_parallel):
        super(GPT, self).__init__()

        self.config = config
        self.world_size = model_parallel

        self.worker_names = [f"worker{r}" for r in range(model_parallel)]
        self.worker_names[0] = "master"

        layers_per_worker = [config.n_layer // model_parallel] * model_parallel
        for i in range(config.n_layer % model_parallel):
            layers_per_worker[i] += 1  # distribute the remainder

        print(f"Distributing {config.n_layer} layers across {model_parallel} workers: {layers_per_worker}")

        self.shard_rrefs = []
        for i in range(model_parallel):
            is_first = (i == 0)
            is_last = (i == model_parallel - 1)
            num_layers = layers_per_worker[i]

            rref = rpc.remote(
                self.worker_names[i],
                GPTPipelineBlock,
                args=(config, is_first, is_last, num_layers)
            )
            self.shard_rrefs.append(rref)
        print("All remote shards created")

        print("Tying embedding and LM head weights...")
        wte_weight_rref = self.shard_rrefs[0].remote().get_wte_weight().to_here()
        self.shard_rrefs[-1].remote().tie_weights(wte_weight_rref)
        print("Weights tied")

    def remote_method(self, method, *args, **kwargs):
        futs = []
        for rref in self.shard_rrefs:
            futs.append(rref.rpc_async().__getattr__(method)(*args, **kwargs))
        torch.futures.wait_all(futs)

    def train(self, mode=True):
        super().train(mode)
        self.remote_method('train', mode)
        return self

    def eval(self):
        return self.train(False)

    def forward(self, idx, targets=None):
        for rref in self.shard_rrefs:
            rref.remote().train(self.training)

        x_rref = RRef(idx)

        for rref in self.shard_rrefs:
            x_rref = rref.remote().forward(x_rref)

        logits = x_rref.to_here()

        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(logits.view(-int(logits.size(0)), -int(logits.size(-1))),
                                               targets.view(-1), ignore_index=-1)

        return logits, loss

    def parameter_rrefs(self):
        remote_params = []
        for rref in self.shard_rrefs:
            remote_params.extend(rref.remote().parameter_rrefs().to_here())
        return remote_params

    def get_num_params(self, non_embedding=True):
        """
        OVERRIDE: Estimate params from config, as they are remote. Cannot just sum on local.
        """
        cfg = self.config
        n_params = (cfg.n_layer * (12 * cfg.n_embd ** 2))
        n_params += (cfg.vocab_size * cfg.n_embd)
        n_params += (cfg.block_size * cfg.n_embd)
        if non_embedding:
            n_params -= (cfg.block_size * cfg.n_embd)
        return n_params