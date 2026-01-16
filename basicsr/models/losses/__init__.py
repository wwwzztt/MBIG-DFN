# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
from .losses import (L1Loss, MSELoss, PSNRLoss, FFTLoss, MultiHeadPSNRLoss, MultiHeadL1Loss_FFTLoss,
                     ONEL1Loss_FFTLoss, MIMOHeadL1Loss_FFTLoss, Stage2_L1Loss_FFTLoss, Stage4_L1Loss_FFTLoss,
                     MIMOOneL1Loss_FFTLoss)

__all__ = [
    'L1Loss', 'MSELoss', 'PSNRLoss', 'FFTLoss', 'MultiHeadPSNRLoss', 'MultiHeadL1Loss_FFTLoss',
    'ONEL1Loss_FFTLoss', 'MIMOHeadL1Loss_FFTLoss', 'Stage2_L1Loss_FFTLoss', 'Stage4_L1Loss_FFTLoss',
    'MIMOOneL1Loss_FFTLoss'
]
