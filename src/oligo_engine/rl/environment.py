"""Gymnasium bridge over the shared fixed-template molecular search space."""

from dataclasses import asdict

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from oligo_engine.design.aso_space import Action, ASODesignSpace
from oligo_engine.reward.scoring import BudgetExhausted, CandidateScore, ScoringSession

OBSERVATION_SCHEMA = "aso-env-v1:onehot19x4,pos,start_delta,remaining,gc,cpg,grun,heads"
ACTION_SCHEMA = ("STOP", "LEFT", "RIGHT")


class ASOEnv(gym.Env):
    """Terminal-only reward; reset/intermediate model scores are always charged."""

    metadata = {"render_modes": []}

    def __init__(self, design_space: ASODesignSpace, session: ScoringSession) -> None:
        super().__init__()
        if design_space.target != session.target:
            raise ValueError("Scorer target and design-space manifest disagree")
        self.design_space, self.session = design_space, session
        self.action_space = spaces.Discrete(3)
        # Each head has bounded utility, available flag and normalized raw value.
        self.observation_space = spaces.Box(
            -1.0,
            1.0,
            shape=(19 * 4 + 6 + 3 * len(session.scorer.manifests),),
            dtype=np.float32,
        )
        self.position = design_space.start
        self.moves = 0
        self.done = True
        self.current_score: CandidateScore | None = None

    def _observation(self) -> np.ndarray:
        candidate = self.design_space.candidate(self.position)
        values = [float(base == b) for base in candidate.sequence for b in "ACGT"]
        extent = max(1, len(self.design_space.target.sequence) - 19)
        values.extend(
            (
                self.position / extent,
                (self.position - self.design_space.start)
                / max(1, self.design_space.radius),
                (self.design_space.horizon - self.moves) / self.design_space.horizon,
                candidate.gc_fraction,
                candidate.cpg_count / 18,
                candidate.longest_g_run / 19,
            )
        )
        for endpoint in self.current_score.endpoints:
            raw = endpoint.raw
            values.extend(
                (
                    endpoint.normalized or 0.0,
                    float(raw is not None),
                    raw / (1 + abs(raw)) if raw is not None else 0.0,
                )
            )
        return np.asarray(values, dtype=np.float32)

    def _info(self, **extra: object) -> dict:
        return {
            "position": self.position,
            "moves": self.moves,
            "score": asdict(self.current_score),
            **extra,
        }

    def action_masks(self) -> np.ndarray:
        return np.asarray(
            (True, False, False)
            if self.done
            else self.design_space.mask(self.position, self.moves),
            dtype=np.bool_,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if options:
            raise ValueError("Starts are fixed by the design-space manifest")
        self.done = True
        score = self.session.score(self.design_space.candidate(self.design_space.start))
        self.position, self.moves = self.design_space.start, 0
        self.current_score, self.done = score, False
        return self._observation(), self._info()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self.done:
            raise RuntimeError("Reset required after termination/truncation")
        if not self.action_space.contains(action) or not self.action_masks()[action]:
            raise ValueError("Illegal or masked action")
        if action != Action.STOP:
            position = self.design_space.transition(self.position, self.moves, action)
            try:
                score = self.session.score(self.design_space.candidate(position))
            except BudgetExhausted:
                # The attempted move did not occur. This is the last valid state,
                # explicitly flagged as budget truncation, never a scored new state.
                self.done = True
                return (
                    self._observation(),
                    0.0,
                    False,
                    True,
                    self._info(
                        reason="budget_exhausted",
                        attempted_position=position,
                        transition_applied=False,
                    ),
                )
            self.position, self.current_score = position, score
            self.moves += 1
        self.done = action == Action.STOP or self.moves == self.design_space.horizon
        reward = self.current_score.terminal_reward() if self.done else 0.0
        return (
            self._observation(),
            reward,
            self.done,
            False,
            self._info(
                transition_applied=True,
            ),
        )
