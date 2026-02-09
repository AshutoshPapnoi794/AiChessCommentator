from collections import deque
from typing import Any, Dict, Optional


class GameNarrativeMemory:
    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.eval_history = deque(maxlen=5)
        self.key_moments = {}
        self.last_commentary = ""
        self.narrative_arc = "The game has just begun."
        self.opening_acknowledged = False
        self.last_opening_announced: Optional[str] = None
        self.pending_forced_response: Optional[Dict[str, Any]] = None
        self.pending_threat_response: Optional[Dict[str, Any]] = None
        self.last_balance_zone: Optional[str] = None
        self.last_momentum_message: str = "None"

    def update_eval(self, cp_value: int, ply: int) -> None:
        self.eval_history.append((ply, cp_value))

    def detect_momentum(self, current_cp: int, is_white_turn: bool) -> str:
        # Mention momentum only on meaningful transitions, not every move.
        balance_zone = "balanced" if abs(current_cp) <= 80 else "imbalanced"
        previous_zone = self.last_balance_zone
        self.last_balance_zone = balance_zone

        if previous_zone is None:
            message = "None" if balance_zone == "balanced" else "A clear imbalance has already appeared."
            if message == self.last_momentum_message:
                return "None"
            self.last_momentum_message = message
            return message

        if previous_zone != "balanced" and balance_zone == "balanced":
            message = "The position has moved back toward balance."
            if message == self.last_momentum_message:
                return "None"
            self.last_momentum_message = message
            return message
        if previous_zone == "balanced" and balance_zone == "balanced":
            return "None"
        if len(self.eval_history) < 2:
            return "None"

        past_cp = [x[1] for x in list(self.eval_history)[:-1]]
        if not past_cp:
            return "None"
        avg_past = sum(past_cp) / len(past_cp)
        diff = current_cp - avg_past

        if not is_white_turn:
            diff = -diff

        if diff > 100:
            message = "Momentum is shifting heavily in favor of the current player."
        elif diff < -100:
            message = "The current player is losing their grip on the position."
        elif abs(diff) < 30:
            message = "None"
        else:
            message = "A gradual positional struggle."

        if message == self.last_momentum_message:
            return "None"
        self.last_momentum_message = message
        return message

    def add_commentary(self, ply: int, text: str) -> None:
        self.last_commentary = text


session_memories: Dict[str, GameNarrativeMemory] = {}
