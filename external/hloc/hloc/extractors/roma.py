import sys
from pathlib import Path
import torch

from ..utils.base_model import BaseModel

sys.path.append(str(Path(__file__).parent / '../../third_party'))

from RoMa.romatch.models.encoders import CNNandDinov2


class RoMa_extractor(BaseModel):

    def _init(self, conf):
        self.net = CNNandDinov2(
            cnn_kwargs=conf.get('cnn_kwargs', dict(pretrained=True, amp=False)),
            amp=conf.get('amp', False),
            dinov2_weights=conf.get('dinov2_weights', None),
            amp_dtype=conf.get('amp_dtype', torch.float16),
        )

    def _forward(self, data):
        return self.net(data['image'])

