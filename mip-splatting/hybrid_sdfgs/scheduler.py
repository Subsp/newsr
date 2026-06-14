from dataclasses import dataclass


@dataclass
class HybridLossScheduler:
    stage_a_end: int
    stage_b_end: int
    stage_a_scale: float = 1.5
    stage_b_scale: float = 1.0
    stage_c_scale: float = 0.35

    def stage(self, iteration: int) -> str:
        if iteration <= self.stage_a_end:
            return "A"
        if iteration <= self.stage_b_end:
            return "B"
        return "C"

    def scale(self, iteration: int) -> float:
        s = self.stage(iteration)
        if s == "A":
            return self.stage_a_scale
        if s == "B":
            return self.stage_b_scale
        return self.stage_c_scale
