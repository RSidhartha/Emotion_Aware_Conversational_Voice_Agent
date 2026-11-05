"""
mood_manager.py
Minimal MoodManager: normalized EMA mood score + threshold + N-turn persistence.
Goes in src/mood/mood_manager.py
"""


class MoodManager:
    def __init__(self, gamma=0.7, threshold=0.5, persistence=2):
        self.gamma = gamma              # decay rate (how much the old mood is kept)
        self.threshold = threshold      # score needed to count as "distressed"
        self.persistence = persistence  # consecutive turns above threshold before switching
        self.score = 0.0
        self.turns_above = 0
        self.mode = "DEFAULT"
        # Inspection state for explain(); written by update(), never read by its math
        self.last_p = None
        self.prev_mode = None
        self.prev_turns_above = None

    def update(self, p_distress: float) -> str:
        self.last_p = p_distress
        self.prev_mode = self.mode
        self.prev_turns_above = self.turns_above

        # 1) EMA update: m_t = gamma * m_{t-1} + (1 - gamma) * p_t
        self.score = self.gamma * self.score + (1 - self.gamma) * p_distress

        # 2) persistence gating
        if self.score >= self.threshold:
            self.turns_above += 1
            if self.turns_above >= self.persistence:
                self.mode = "EMPATHETIC"
        else:
            self.turns_above = 0
            self.mode = "DEFAULT"
        return self.mode

    def explain(self) -> str:
        """One-line reason for the mode chosen by the last update()."""
        if self.last_p is None:
            return "no update yet -> DEFAULT"
        thr = f"threshold {self.threshold:.2f}"
        if self.score >= self.threshold:
            n, k = self.turns_above, self.persistence
            turns = f"for {n}/{k} turns" if n <= k else f"for {n} turns (>= {k})"
            reason = f"score {self.score:.2f} >= {thr} {turns} -> {self.mode}"
            if n < k:
                reason += f" (needs {k} consecutive turns)"
        else:
            reason = f"score {self.score:.2f} < {thr} -> {self.mode}"
            if self.prev_turns_above:
                reason += " (turns_above reset)"
        if self.mode != self.prev_mode:
            reason += f" [switched from {self.prev_mode}]"
        return reason


def reactive_mode(p_distress: float, threshold=0.5) -> str:
    """Baseline: react to each turn in isolation, no memory of previous turns."""
    return "EMPATHETIC" if p_distress >= threshold else "DEFAULT"