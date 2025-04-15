import copy

import cheetah
import torch
import torch.nn as nn


class BeamDynamics(nn.Module):
    def __init__(
        self,
        segment,
        incoming_parameters,
    ):
        super().__init__()
        self.segment = copy.deepcopy(segment)

        # Store incoming_parameters as a Parameter to ensure it stays in the graph
        self.incoming_params = copy.deepcopy(incoming_parameters)
        self.device = self.incoming_params.device

        # Move entire module to device
        self.to(self.device)

        # Create beam
        self.incoming = cheetah.ParameterBeam.from_parameters(
            energy=self.incoming_params[0],
            mu_x=self.incoming_params[1],
            mu_px=self.incoming_params[2],
            mu_y=self.incoming_params[3],
            mu_py=self.incoming_params[4],
            sigma_x=self.incoming_params[5],
            sigma_px=self.incoming_params[6],
            sigma_y=self.incoming_params[7],
            sigma_py=self.incoming_params[8],
            sigma_tau=self.incoming_params[9],
            sigma_p=self.incoming_params[10],
            dtype=torch.float32,
        )

    def forward(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        # Ensure 2D batch dimension
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if u.dim() == 1:
            u = u.unsqueeze(0)

        x = x.clone().requires_grad_(True).to(self.device)
        u = u.clone().requires_grad_(True).to(self.device)

        #mu_x = torch.as_tensor(x[0, 0], dtype=x.dtype, device=x.device)
        #mu_x.requires_grad_(True)
        #mu_y = torch.as_tensor(x[0, 1], dtype=x.dtype, device=x.device)
        #mu_y.requires_grad_(True)
        #sigma_x = torch.as_tensor(x[0, 2], dtype=x.dtype, device=x.device)
        #sigma_x.requires_grad_(True)
        #sigma_y = torch.as_tensor(x[0, 3], dtype=x.dtype, device=x.device)
        #sigma_y.requires_grad_(True)

        # Extract parameters with batch dimension
        mu_x = x[:, 0].clone().requires_grad_(True)  # Shape: [batch_size]
        mu_y = x[:, 1].clone().requires_grad_(True)
        sigma_x = x[:, 2].clone().requires_grad_(True)
        sigma_y = x[:, 3].clone().requires_grad_(True)

        # Create beam with batched parameters
        beam = self.incoming.transformed_to(
            energy=self.incoming_params[0],  # Fixed, no gradients needed
            mu_x=mu_x,  # Dynamic, batched
            mu_px=self.incoming_params[2],  # Fixed, no gradients needed
            mu_y=mu_y,  # Dynamic, batched
            mu_py=self.incoming_params[4],  # Fixed, no gradients needed
            sigma_x=sigma_x,  # Dynamic, batched
            sigma_px=self.incoming_params[6],  # Fixed, no gradients needed
            sigma_y=sigma_y,  # Dynamic, batched
            sigma_py=self.incoming_params[8],  # Fixed, no gradients needed
            sigma_tau=self.incoming_params[9],  # Fixed, no gradients needed
            sigma_p=self.incoming_params[10],  # Fixed, no gradients needed
        )

        # Track beam
        next_beam = self.segment.track(beam)

        # Extract next state, preserving batch dimension
        x_next = torch.stack(
            [
                next_beam.mu_x, #.clone().requires_grad_(True),
                next_beam.mu_y, #.clone().requires_grad_(True),
                next_beam.sigma_x, #.clone().requires_grad_(True),
                next_beam.sigma_y, #.clone().requires_grad_(True),
            ],
            dim=-1,
        ).to(dtype=torch.float32)  # Shape: [batch_size, 4]

        return x_next

    def _set_magnets(self, u: torch.Tensor) -> None:
        """
        Set magnet parameters based on control input u.
        """
        # Direct assignment to preserve u's graph and  gradients
        self.segment.AREAMQZM1.k1 = u[:, 0].clone().requires_grad_(True)
        self.segment.AREAMQZM2.k1 = u[:, 1].clone().requires_grad_(True)
        self.segment.AREAMCVM1.angle = u[:, 2].clone().requires_grad_(True)
        self.segment.AREAMQZM3.k1 = u[:, 3].clone().requires_grad_(True)
        self.segment.AREAMCHM1.angle = u[:, 4].clone().requires_grad_(True)
