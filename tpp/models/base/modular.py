import torch as th
import torch.nn as nn
import torch.nn.functional as F
from tpp.utils.events import Events

from typing import Dict, Optional, Tuple

from tpp.models.base.enc_dec import EncDecProcess


class ModularProcess(EncDecProcess):
    """Build a process out of multiple process instances.

    Args:
        processes: A list of instances of EncDecProcess with the same number
            of marks.
        use_coefficients: If `True` puts a learnable positive coefficient in
            front of each process, i.e. lambda = sum_i alpha_i lambda_i.
            Defaults to `False`.
    """
    def __init__(
            self,
            processes: Dict[str, EncDecProcess],
            use_coefficients: Optional[bool] = False,
            multi_labels: Optional[bool] = False,
            **kwargs):
        name = '_'.join([p.name for p in processes.values()])

        marks = {p.marks for p in processes.values()}
        if len(marks) > 1:
            raise ValueError("The number of independent marks ({}) is {}. It "
                             "should be 1.".format(marks, len(marks)))
        marks = list(marks)[0]

        super(ModularProcess, self).__init__(
            name=name, marks=marks,
            encoder=None, decoder=None, multi_labels=multi_labels, **kwargs)

        self.processes = processes
        for k, p in self.processes.items():
            self.add_module(k, p)
        self.n_processes = len(processes)
        self.use_coefficients = use_coefficients
        if self.use_coefficients:
            self.alpha = nn.Parameter(th.Tensor(self.n_processes))
            self.reset_parameters()

    def reset_parameters(self):
        if "poisson" in self.processes:
            init_constant = [
                1. if x == "poisson" else -2. for x in self.processes]
            init_constant = th.Tensor(init_constant).to(
                self.alpha.device).type(
                self.alpha.dtype)
            self.alpha.data = init_constant
        else:
            nn.init.uniform_(self.alpha)

    def artifacts(
            self, query: th.Tensor, events: Events
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, Dict]:
        """Compute the (log) intensities and intensity integrals at query times
        given events.

        Args:
            query: [B,T] Sequences of query times to evaluate the intensity
                function.
            events: [B,L] Times and labels of events.

        Returns:
            log_intensity: [B,T,M] The log intensities for each query time for
                each mark (class).
            intensity_integral: [B,T,M] The integral of the intensity from
                the most recent event to the query time for each mark.
            intensities_mask: [B,T,M] Which intensities are valid for further
                computation based on e.g. sufficient history available.
            artifacts: A dictionary of whatever else you might want to return.

        """
        # [B,T,M], [B,T,M], [B,T], Dict
        artifacts = {
            k: p.artifacts(query=query, events=events)
            for k, p in self.processes.items()}

        log_intensity, intensity_integral, intensity_mask, _ = zip(
            *artifacts.values())

        log_intensity = th.stack(log_intensity, dim=0)              # [P,B,T,M]
        intensities_mask = th.stack(intensity_mask, dim=0)          # [P,B,T,M]
        intensity_integral = th.stack(intensity_integral, dim=0)    # [P,B,T,M]

        if self.use_coefficients:
            alpha = self.alpha.reshape(-1, 1, 1, 1)                 # [P,1,1,1]
            alpha = F.softmax(alpha, dim=0)
            log_alpha = th.log(alpha)
            log_intensity = log_intensity + log_alpha               # [P,B,T,M]

            intensity_integral = alpha * intensity_integral         # [P,B,T,M]

        log_intensity = th.logsumexp(log_intensity, dim=0)          # [B,T,M]
        intensity_integral = th.sum(intensity_integral, dim=0)      # [B,T,M]
        intensities_mask = th.prod(intensities_mask, dim=0)         # [B,T,M]

        artifacts = {k: v[-1] for k, v in artifacts.items()}

        return log_intensity, intensity_integral, intensities_mask, artifacts

    def encode(self, **kwargs):
        pass

    def decode(self, **kwargs):
        pass
