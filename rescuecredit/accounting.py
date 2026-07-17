from dataclasses import dataclass


@dataclass
class BudgetCounter:
    main_steps: int = 0
    shadow_steps: int = 0
    failed_replay_steps: int = 0
    evaluation_steps: int = 0

    @property
    def training_steps(self) -> int:
        return self.main_steps + self.shadow_steps + self.failed_replay_steps

    @property
    def total_steps(self) -> int:
        return self.training_steps + self.evaluation_steps

    def charge_main(self, steps: int = 1) -> None:
        self.main_steps += self._checked(steps)

    def charge_shadow(self, steps: int = 1) -> None:
        self.shadow_steps += self._checked(steps)

    def charge_failed_replay(self, steps: int = 1) -> None:
        self.failed_replay_steps += self._checked(steps)

    def charge_evaluation(self, steps: int = 1) -> None:
        self.evaluation_steps += self._checked(steps)

    @staticmethod
    def _checked(steps: int) -> int:
        if steps < 0:
            raise ValueError("step charge cannot be negative")
        return int(steps)

