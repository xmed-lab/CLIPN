import torch
import torch.nn as nn
from torch.nn import functional as F

try:
    import torch.distributed.nn
    from torch import distributed as dist
    has_distributed = True
except ImportError:
    has_distributed = False

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


def gather_features(
        image_features,
        text_features,
        text_features_no,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        use_horovod=False
):
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    if use_horovod:
        assert hvd is not None, 'Please install horovod'
        if gather_with_grad:
            all_image_features = hvd.allgather(image_features)
            #all_image_features_no = hvd.allgather(image_features_no)
            all_text_features = hvd.allgather(text_features)
            all_text_features_no = hvd.allgather(text_features_no)
        else:
            with torch.no_grad():
                all_image_features = hvd.allgather(image_features)
                #all_image_features_no = hvd.allgather(image_features_no)
                all_text_features = hvd.allgather(text_features)
                all_text_features_no = hvd.allgather(text_features_no)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features = list(all_image_features.chunk(world_size, dim=0))
                #gathered_image_features_no = list(all_image_features_no.chunk(world_size, dim=0))
                gathered_text_features = list(all_text_features.chunk(world_size, dim=0))
                gathered_text_features_no = list(all_text_features_no.chunk(world_size, dim=0))
                gathered_image_features[rank] = image_features
                #gathered_image_features_no[rank] = image_features_no
                gathered_text_features[rank] = text_features
                gathered_text_features_no[rank] = text_features_no
                all_image_features = torch.cat(gathered_image_features, dim=0)
                #all_image_features_no = torch.cat(gathered_image_features_no, dim=0)
                all_text_features = torch.cat(gathered_text_features, dim=0)
                all_text_features_no = torch.cat(gathered_text_features_no, dim=0)
    else:
        # We gather tensors from all gpus
        if gather_with_grad:
            all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
            #all_image_features_no = torch.cat(torch.distributed.nn.all_gather(image_features_no), dim=0)
            all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
            all_text_features_no = torch.cat(torch.distributed.nn.all_gather(text_features_no), dim=0)
        else:
            gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
            #gathered_image_features_no = [torch.zeros_like(image_features_no) for _ in range(world_size)]
            gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
            gathered_text_features_no = [torch.zeros_like(text_features_no) for _ in range(world_size)]
            dist.all_gather(gathered_image_features, image_features)
            #dist.all_gather(gathered_image_features_no, image_features_no)
            dist.all_gather(gathered_text_features, text_features)
            dist.all_gather(gathered_text_features_no, text_features_no)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features[rank] = image_features
                #gathered_image_features_no[rank] = image_features_no
                gathered_text_features[rank] = text_features
                gathered_text_features_no[rank] = text_features_no
            all_image_features = torch.cat(gathered_image_features, dim=0)
            #all_image_features_no = torch.cat(gathered_image_features_no, dim=0)
            all_text_features = torch.cat(gathered_text_features, dim=0)
            all_text_features_no = torch.cat(gathered_text_features_no, dim=0)

    return all_image_features, all_text_features, all_text_features_no


class ClipLoss(nn.Module):

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}
        self.eyes = {}

    def forward(self, image_features, text_features, text_features_no, logit_scale):
        device = image_features.device
        if self.world_size > 1:
            all_image_features, all_text_features, all_text_features_no = gather_features(
                image_features, text_features, text_features_no,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
                logits_per_text_no = logit_scale * text_features_no @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
                logits_per_image_no = logit_scale * all_image_features @ all_text_features_no.T
                #logits_intra_no = logit_scale * all_text_features_no @ all_text_features_no.T
                    
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T

        # calculated ground-truth and cache if enabled
        num_logits = logits_per_image.shape[0]
        
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            eyes = torch.eye(num_logits, device=device)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.eyes[device] = eyes
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
            eyes = self.eyes[device]
        
        logits_per_image_yes_no = torch.cat([logits_per_image.unsqueeze(-1), logits_per_image_no.unsqueeze(-1)], dim=-1)
        logits_per_image_yes_no = F.softmax(logits_per_image_yes_no, dim=-1)
        
        loss_bin_yes, loss_bin_no = self.image_text_binary_opposite_loss(logits_per_image_yes_no, eyes)
        loss_tso = self.text_semantic_opposite_loss(all_text_features, all_text_features_no)

        return loss_bin_yes, loss_bin_no, loss_tso
    def text_semantic_opposite_loss(self, all_text_features, all_text_features_no, mode="L2"):
        if mode == "L2":
            l2_distance = 2 - 2 * (all_text_features * all_text_features_no).sum(-1) + 1e-4# epsilon = 1e-4, used to get rid of inifity gradient
            loss = 2 - torch.sqrt(l2_distance) # \in [0,2]
        if mode == "cosine":
            loss = (all_text_features * all_text_features_no).sum(-1)  + 1.0  # \in [0,2]
        return loss.mean()

    def image_text_binary_opposite_loss(self, logits_per_image_yes_no, eyes):
        N = logits_per_image_yes_no.shape[0]
        binary_yes_no = eyes * logits_per_image_yes_no[:,:,0] + (1-eyes) * logits_per_image_yes_no[:,:,1]
        loss_bin = - torch.log(binary_yes_no) 
            
        loss_bin_yes = (eyes * loss_bin).view(-1).sum() / N
        loss_bin_no = ((1-eyes) * loss_bin).view(-1).sum() / (N**2 - N)
            
        return loss_bin_yes, loss_bin_no
        
    
