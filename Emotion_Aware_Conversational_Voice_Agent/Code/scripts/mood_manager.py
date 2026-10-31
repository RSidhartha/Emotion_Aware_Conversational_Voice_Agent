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

    def update(self, p_distress: float) -> str:
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


def reactive_mode(p_distress: float, threshold=0.5) -> str:
    """Baseline: react to each turn in isolation, no memory of previous turns."""
    return "EMPATHETIC" if p_distress >= threshold else "DEFAULT"