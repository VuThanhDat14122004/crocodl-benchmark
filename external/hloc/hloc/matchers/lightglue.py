from ..utils.base_model import BaseModel
from lightglue import LightGlue as LightGlue_

class LightGlue(BaseModel):
    default_conf = {
        'features': 'superpoint',
        'depth_confidence': 0.95,
        'width_confidence': 0.99,
    }
    required_inputs = [
        'image0', 'keypoints0', 'descriptors0',
        'image1', 'keypoints1', 'descriptors1',
    ]

    def _init(self, conf):
        self.net = LightGlue_(conf.pop('features'), **conf)

    def _forward(self, data):
        # print(f"LightGlue input keys: {data.keys()}")
        data['descriptors0'] = data['descriptors0'].transpose(-1, -2)
        data['descriptors1'] = data['descriptors1'].transpose(-1, -2)
        # check = {
        #     'image0': {k[:-1]: v for k, v in data.items() if k[-1] == '0'},
        #     'image1': {k[:-1]: v for k, v in data.items() if k[-1] == '1'}
        # }
        # print(f"check keys: {check.keys()}")
        # for k,v in data.items():
        #     try:
        #         print(f"K :{k}; V shape: {v.shape}")
        #     except:
        #         print(f"K :{k}; V type: {type(v)}")
        print(f"size of image0: {data['image_size0']}, size of image1: {data['image_size1']}")
        return self.net({
            'image0': {k[:-1]: v for k, v in data.items() if k[-1] == '0'},
            'image1': {k[:-1]: v for k, v in data.items() if k[-1] == '1'}
        })
