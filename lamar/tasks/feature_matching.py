import logging
from typing import Optional
from copy import deepcopy

from hloc import match_features

from .feature_extraction import FeatureExtraction
from .pair_selection import PairSelection
from ..utils.misc import same_configs, write_config

logger = logging.getLogger(__name__)


class FeatureMatchingPaths:
    def __init__(self, root, config, query_id, ref_id):
        self.root = root
        feature_name = config['features']['name']
        matches_name = config['name']
        self.workdir = root / 'matching' / query_id / ref_id / feature_name / matches_name
        self.matches = self.workdir / 'matches.h5'
        self.config = self.workdir / 'configuration.json'

class FeatureMatching:
    methods = {
        'superglue': {
            'name': 'superglue',
            'hloc': {
                'model': {
                    'name': 'superglue',
                    'weights': 'outdoor',
                    'sinkhorn_iterations': 5,
                },
            },
        },
        'lightglue': {
            'name': 'lightglue',
            'hloc': {
                'model': {
                    'name': 'lightglue',
                    'features': 'superpoint',
                },
            },
        },
        'mnn': {
            'name': 'mnn',
            'hloc': {
                'model': {
                    'name': 'nearest_neighbor',
                    'do_mutual_check': True,
                },
            }
        },
        'ratio_mnn_0_9': {
            'name': 'ratio_mnn',
            'hloc': {
                'model': {
                    'name': 'nearest_neighbor',
                    'do_mutual_check': True,
                    'ratio_threshold': 0.9,
                },
            }
        },
        'ratio_mnn_0_8': {
            'name': 'ratio_mnn',
            'hloc': {
                'model': {
                    'name': 'nearest_neighbor',
                    'do_mutual_check': True,
                    'ratio_threshold': 0.8,
                },
            }
        },
        'adalam': {
            'name': 'adalam',
            'hloc': {
                'model': {
                    'name': 'adalam'
                },
            }
        },
        # lightglue_gim matcher
        'lightglue_gim': {
            'name': 'lightglue_gim',
            'hloc': {
                'model': {
                    'name': 'lightglue_gim',
                    'features': 'superpoint',
                },
            },
        },
        ## lightglue+roma matcher
        'lightglue+roma': {
            'name': 'lightglue+roma',
            'hloc': {
                'output': 'matches-superglue-roma',
                'model': {
                    'name': 'roma',
                    'max_keypoints': 4096,
                    'weight_mode': 'indoor',
                    'resize_max': 1024,
                    'dist_threshold': 5.0,
                },
                'model2': {
                    'name': 'lightglue_gim',
                    'features': 'superpoint',
                    'preprocessing': {
                        'resize_max': 1024,
                        'resize_force': True,
                    },
                }
            }
        }
    }
                     

    def __init__(self, outputs, capture, query_id, ref_id, config,
                 pair_selection: PairSelection,
                 extraction: FeatureExtraction,
                 extraction_ref: Optional[FeatureExtraction] = None,
                 overwrite=False, is_query_map=False,feature_path_raw_ref: Optional[str]=None):
        
        extraction_ref = extraction_ref or extraction
        if extraction.config['name'] != extraction_ref.config['name']:
            raise ValueError('Matching two different features: '
                             f'{extraction.config} vs {extraction_ref.config}')
        assert query_id == extraction.session_id
        assert query_id == pair_selection.query_id
        assert ref_id == extraction_ref.session_id
        assert ref_id == pair_selection.ref_id

        self.config = config = {
            **deepcopy(config),
            'features': extraction.config,  # detect upstream changes
            # do not include the pairs so the same file can be reused
        }
            
        self.query_id = query_id
        self.ref_id = ref_id
        self.extraction = extraction
        self.extraction_ref = extraction_ref
        self.pair_selection = pair_selection
        self.paths = FeatureMatchingPaths(outputs, config, query_id, ref_id)
        self.paths.workdir.mkdir(parents=True, exist_ok=True)

        logger.info('Matching local features with %s for sessions (%s, %s).',
                    config['name'], query_id, ref_id)
        if not same_configs(config, self.paths.config):
            logger.warning('Existing matches will be overwritten.')
            overwrite = True
        print(f'\n----- overwrite matching: {overwrite} -----\n')
        # overwrite = False
        if is_query_map and 'roma' in config['hloc']['model']['name']:
            print(f'start query-map roma matching...')
            match_features.main(
                config['hloc'],
                pair_selection.paths.pairs_hloc,
                extraction.paths.features,
                matches=self.paths.matches,
                features_ref=extraction_ref.paths.features,
                overwrite=overwrite,
                dict_keypoints_index_query=extraction.paths.dict_keypoints_index,
                dict_keypoints_index_map=extraction_ref.paths.dict_keypoints_index,
                is_query_map_match=True,
                feature_path_raw_ref=feature_path_raw_ref
            )
        elif 'roma' in config['hloc']['model']['name']:
            print(f'start map-map roma matching...')
            match_features.main(
                config['hloc'],
                pair_selection.paths.pairs_hloc,
                extraction.paths.features,
                matches=self.paths.matches,
                features_ref=extraction_ref.paths.features,
                overwrite=overwrite,
                dict_keypoints_index_query=extraction.paths.dict_keypoints_index,
                dict_keypoints_index_map=extraction_ref.paths.dict_keypoints_index,
            )
        else:
            match_features.main(
            config['hloc'],
            pair_selection.paths.pairs_hloc,
            extraction.paths.features,
            matches=self.paths.matches,
            features_ref=extraction_ref.paths.features,
            overwrite=overwrite,
            )

        write_config(config, self.paths.config)
