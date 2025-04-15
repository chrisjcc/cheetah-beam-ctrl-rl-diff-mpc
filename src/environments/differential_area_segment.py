from pathlib import Path

import cheetah
import torch.nn as nn


class DifferentialAREASegment(nn.Module):
    def __init__(self):
        super().__init__()
        # Load lattice segment from JSON and select subcell
        self.segment = cheetah.Segment.from_lattice_json(
            Path(__file__).parent / "ea.json"
        )

        # Enable screen for beam measurements
        self.segment.AREABSCR1.is_active = True

        # Forward element attributes to self
        for element in self.segment.elements:
            setattr(self, element.name, element)

        # Forward track method to self
        setattr(self, "track", self.segment.track)

    def forward(self, incoming_beam: cheetah.Beam):
        # Track beam through segment (differentiable operation)
        # Rely on external magnet settings (e.g., from BeamDynamics)
        # BeamDynamics._set_magnets handles all magnet updates, and
        # DifferentialAREASegment.forward just tracks the beam.
        outgoing_beam = self.segment.track(incoming_beam)

        return outgoing_beam
