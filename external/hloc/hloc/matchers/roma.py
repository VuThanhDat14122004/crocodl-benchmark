import sys
from pathlib import Path
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.base_model import BaseModel

sys.path.append(str(Path(__file__).parent / '../../third_party'))
from RoMa.romatch.models.matcher import (
    ConvRefiner,
    CosKernel,
    GP,
    Decoder,
)
from third_party.RoMa.romatch.utils.kde import kde
# from ...third_party.RoMa.romatch.utils.kde import kde
#from RoMa.romatch.models.encoders import CNNandDinov2
from RoMa.romatch.models.transformer import Block, TransformerDecoder, MemEffAttention

class RoMa_matcher(BaseModel):

    def _init(self, conf):
        gp_dim = 512
        feat_dim = 512
        decoder_dim = gp_dim + feat_dim
        cls_to_coord_res = 64
        kernel = CosKernel
        kernel_temperature = 0.2
        learn_temperature = False
        only_attention = False
        basis = "fourier"
        no_cov = True
        gp16 = GP(
            kernel,
            T=kernel_temperature,
            learn_temperature=learn_temperature,
            only_attention=only_attention,
            gp_dim=gp_dim,
            basis=basis,
            no_cov=no_cov,
        )
        gps = nn.ModuleDict({"16": gp16})
        coordinate_decoder = TransformerDecoder(
            nn.Sequential(
                *[Block(decoder_dim, 8, attn_class=MemEffAttention) for _ in range(5)]
            ),
            decoder_dim,
            cls_to_coord_res**2 + 1,
            is_classifier=True,
            amp=True,
            pos_enc=False,
        )
        proj16 = nn.Sequential(nn.Conv2d(1024, 512, 1, 1), nn.BatchNorm2d(512))
        proj8 = nn.Sequential(nn.Conv2d(512, 512, 1, 1), nn.BatchNorm2d(512))
        proj4 = nn.Sequential(nn.Conv2d(256, 256, 1, 1), nn.BatchNorm2d(256))
        proj2 = nn.Sequential(nn.Conv2d(128, 64, 1, 1), nn.BatchNorm2d(64))
        proj1 = nn.Sequential(nn.Conv2d(64, 9, 1, 1), nn.BatchNorm2d(9))
        proj = nn.ModuleDict(
            {
                "16": proj16,
                "8": proj8,
                "4": proj4,
                "2": proj2,
                "1": proj1,
            }
        )
        dw = True
        hidden_blocks = 8
        kernel_size = 5
        displacement_emb = "linear"
        disable_local_corr_grad = True
        use_custom_corr = False
        partial_conv_refiner = partial(
            ConvRefiner,
            kernel_size=kernel_size,
            dw=dw,
            hidden_blocks=hidden_blocks,
            displacement_emb=displacement_emb,
            corr_in_other=True,
            amp=True,
            disable_local_corr_grad=disable_local_corr_grad,
            bn_momentum=0.01,
            use_custom_corr=use_custom_corr,
        )
        conv_refiner = nn.ModuleDict(
            {
                "16": partial_conv_refiner(
                    2 * 512 + 128 + (2 * 7 + 1) ** 2,
                    2 * 512 + 128 + (2 * 7 + 1) ** 2,
                    2 + 1,
                    displacement_emb_dim=128,
                    local_corr_radius=7,
                ),
                "8": partial_conv_refiner(
                    2 * 512 + 64 + (2 * 3 + 1) ** 2,
                    2 * 512 + 64 + (2 * 3 + 1) ** 2,
                    2 + 1,
                    displacement_emb_dim=64,
                    local_corr_radius=3,
                ),
                "4": partial_conv_refiner(
                    2 * 256 + 32 + (2 * 2 + 1) ** 2,
                    2 * 256 + 32 + (2 * 2 + 1) ** 2,
                    2 + 1,
                    displacement_emb_dim=32,
                    local_corr_radius=2,
                ),
                "2": partial_conv_refiner(
                    2 * 64 + 16,
                    128 + 16,
                    2 + 1,
                    displacement_emb_dim=16,
                ),
                "1": partial_conv_refiner(
                    2 * 9 + 6,
                    24,
                    2 + 1,
                    displacement_emb_dim=6,
                ),
            }
        )
        displacement_dropout_p = 0.0
        gm_warp_dropout_p = 0.0
        self.decoder = Decoder(
            coordinate_decoder,
            gps,
            proj,
            conv_refiner,
            detach=True,
            scales=["16", "8", "4", "2", "1"],
            displacement_dropout_p=displacement_dropout_p,
            gm_warp_dropout_p=gm_warp_dropout_p,
        ) 
        self.net = self.decoder
        self.finest_scale = 1 # conf['finest_scale']
        self.attenuate_cert = True
        self.sample_mode = "threshold_balanced"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sample_thresh = 100
        self.max_keypoints = conf['max_keypoints']
        # self.resolution=(14 * 8 * 6, 14 * 8 * 6)
    
    
    def to_pixel_coordinates(self, coords, H_A, W_A, H_B=None, W_B=None):
        if coords.shape[-1] == 2:
            return self._to_pixel_coordinates(coords, H_A, W_A)

        if isinstance(coords, (list, tuple)):
            kpts_A, kpts_B = coords[0], coords[1]
        else:
            kpts_A, kpts_B = coords[..., :2], coords[..., 2:]
        return self._to_pixel_coordinates(kpts_A, H_A, W_A), self._to_pixel_coordinates(
            kpts_B, H_B, W_B
        )

    def _to_pixel_coordinates(self, coords, H, W):
        kpts = torch.stack(
            (W / 2 * (coords[..., 0] + 1), H / 2 * (coords[..., 1] + 1)), axis=-1
        )
        return kpts
    
    def _forward(self, data):
        for key in data[0].keys(): # data[0] is dict: {1: tensor(1,64, H_resize, W_resize), 2: tensor, 4: tensor, 8: tensor, 16: tensor(1,1024, H/14, W/14), image_size: (H_origin,W_origin)}
            if key != 'image_size':
                print(f"\ndata[0][{key}].shape: {data[0][key].shape}")
        print(f"\ndata[0]['image_size']: {data[0]['image_size']}")
        corresps = self.net(data)
        # print shape before decoder
        print(f"\n----------decoder corresps keys:{ corresps.keys()}---------------\n")
        # corresps is dict: {1: {'flow': tensor(1, 2, H, W), 'certainty': tensor(1, 1, H, W)}, 2: {...}, 4: {...}, 8: {...}, 16: {...}}
        for key in corresps.keys():
            print(f"\n-scale {key} : flow {corresps[key]['flow'].shape}, certainty {corresps[key]['certainty'].shape}\n")
        b = 1 # fixed_batch_size
        H_A, W_A = data[0][1].shape[-2:]
        H_B, W_B = data[1][1].shape[-2:]
        assert H_A == H_B and W_A == W_B, "Input images must have the same spatial dimensions"
        hs, ws = H_A, W_A
        if self.attenuate_cert:
            low_res_certainty = F.interpolate(
                corresps[16]["certainty"],
                size=(hs, ws),
                align_corners=False,
                mode="bilinear",
            )
            cert_clamp = 0
            factor = 0.5
            low_res_certainty = (
                factor * low_res_certainty * (low_res_certainty < cert_clamp)
            )
        im_A_to_im_B = corresps[self.finest_scale]["flow"]
        certainty = corresps[self.finest_scale]["certainty"] - (
            low_res_certainty if self.attenuate_cert else 0
        )
        im_A_to_im_B = im_A_to_im_B.permute(0, 2, 3, 1)
        # Create im_A meshgrid
        im_A_coords = torch.meshgrid(
            (
                torch.linspace(-1 + 1 / hs, 1 - 1 / hs, hs, device=self.device),
                torch.linspace(-1 + 1 / ws, 1 - 1 / ws, ws, device=self.device),
            ),
            indexing="ij",
        )
        im_A_coords = torch.stack((im_A_coords[1], im_A_coords[0]))
        im_A_coords = im_A_coords[None].expand(b, 2, hs, ws)
        certainty = certainty.sigmoid()  # logits -> probs
        im_A_coords = im_A_coords.permute(0, 2, 3, 1)
        if (im_A_to_im_B.abs() > 1).any() and True:
            wrong = (im_A_to_im_B.abs() > 1).sum(dim=-1) > 0
            certainty[wrong[:, None]] = 0
        im_A_to_im_B = torch.clamp(im_A_to_im_B, -1, 1)
        warp = torch.cat((im_A_coords, im_A_to_im_B), dim=-1)
        
        #sample
        matches = warp
        certainty = certainty[:, 0]
        num_max_keypoints = self.max_keypoints
        
        if "threshold" in self.sample_mode:
            upper_thresh = self.sample_thresh
            certainty = certainty.clone()
            certainty[certainty > upper_thresh] = 1
        matches, certainty = (
            matches.reshape(-1, 4),
            certainty.reshape(-1),
        )
        expansion_factor = 4 if "balanced" in self.sample_mode else 1
        good_samples = torch.multinomial(
            certainty,
            num_samples=min(expansion_factor * num_max_keypoints, len(certainty)),
            replacement=False,
        )
        good_matches, good_certainty = matches[good_samples], certainty[good_samples]
        if "balanced" not in self.sample_mode:
            return good_matches, good_certainty
        density = kde(good_matches, std=0.1)
        p = 1 / (density + 1)
        p[density < 10] = (
            1e-7  # Basically should have at least 10 perfect neighbours, or around 100 ok ones
        )
        balanced_samples = torch.multinomial(
            p, num_samples=min(num_max_keypoints, len(good_certainty)), replacement=False
        )
        matches, certainty = good_matches[balanced_samples], good_certainty[balanced_samples]

        # convert to pixel coordinates
        kpts1, kpts2 = self.to_pixel_coordinates(
            matches, H_A, W_A, H_B, W_B
        ) 
        
        pred = {
            "keypoints0": kpts1,
            "keypoints1": kpts2,
            "mconf": certainty,
        }

        return pred


