from tpp.models.base.enc_dec import EncDecProcess
from tpp.models.encoders.stub import StubEncoder
from tpp.models.decoders.hawkes import HawkesDecoder

from typing import Optional


class HawkesProcess(EncDecProcess):
    def __init__(
            self,
            multi_labels: Optional[bool] = False,
            marks: Optional[int] = 1,
            **kwargs):
        super(HawkesProcess, self).__init__(
            encoder=StubEncoder(marks=marks),
            decoder=HawkesDecoder(marks=marks),
            multi_labels=multi_labels)
        self.alpha = self.decoder.alpha
        self.beta = self.decoder.beta
        self.mu = self.decoder.mu
