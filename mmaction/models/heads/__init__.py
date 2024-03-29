from .audio_tsn_head import AudioTSNHead
from .base import BaseHead
from .i3d_head import I3DHead
from .slowfast_head import SlowFastHead
from .ssn_head import SSNHead
from .tpn_head import TPNHead
from .tsm_head import TSMHead
from .tsn_head import TSNHead
from .x3d_head import X3DHead
from .fusion_head import FusionHead
from .full_trailer_cnn_head import FullTrailerCNNHead
from .full_trailer_maxpool_head import FullTrailerMaxPoolHead
from .full_trailer_regression_head import FullTrailerRegressionHead
__all__ = [
    'TSNHead', 'I3DHead', 'BaseHead', 'TSMHead', 'SlowFastHead', 'SSNHead',
    'TPNHead', 'AudioTSNHead', 'X3DHead','FusionHead','FullTrailerCNNHead','FullTrailerMaxPoolHead','FullTrailerRegressionHead',
]
