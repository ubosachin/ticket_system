from typing import Any
from openenv.core.rubrics import Rubric

# Score bounds — must be strictly between 0 and 1 per platform requirements
SCORE_MIN = 0.01  # Never exactly 0.0
SCORE_MAX = 0.99  # Never exactly 1.0

def clamp_score(s: float) -> float:
    """Clamp score to strictly-valid (0, 1) range with epsilon margins."""
    return max(0.01, min(0.99, round(s, 4)))

class TicketSystemRubric(Rubric):
    """
    Rubric for the Ticket System environment.

    Tracks cumulative reward across a trajectory.
    Starting reward (set at reset): 0.2
    Maximum achievable reward:      0.72

    All returned per-step rewards are non-negative (never < 0).
    Cumulative reward is strictly bounded within (SCORE_MIN, SCORE_MAX).
    """

    def __init__(self):
        super().__init__()
        self.reset()

    def reset(self):
        self.current_reward = 0.5          # start in middle of valid range
        self.refund_issued = False
        self.ticket_resolved = False
        self.read_ticket_rewarded = False
        self.search_orders_rewarded = False
        self.get_order_status_rewarded = False
        # Seed last_score with clamped value
        object.__setattr__(self, "last_score", clamp_score(0.5))
        object.__setattr__(self, "score", clamp_score(0.5))
    
    def get_score(self) -> float:
        """Return the current cumulative score."""
        return self.current_reward

    def _clamp(self, value: float) -> float:
        """Clamp value to strictly-valid range."""
        return min(max(value, 0.0), SCORE_MAX - self.current_reward)

    def forward(self, action: Any, observation: Any) -> float:
        """
        Compute incremental reward for a single step.

        Returns a NON-NEGATIVE float. Cumulative reward stays strictly within (0.15, 0.85).
        """
        task_name = observation.metadata.get("task", "easy")
        action_type = action.action_type
        msg = getattr(action, "message", "") or ""
        msg_lower = msg.lower()

        reward = 0.0

        # Once ticket is resolved, no more rewards
        if self.ticket_resolved:
            return 0.0

        if action_type == "read_ticket" and not self.read_ticket_rewarded:
            reward = 0.05
            self.read_ticket_rewarded = True

        elif action_type == "search_orders" and not self.search_orders_rewarded:
            order_found = (
                observation.orders_found
                and observation.orders_found != "No orders found."
            )
            if order_found and task_name in [
                "medium", "ticket_medium", "hard", "ticket_hard"
            ]:
                reward = 0.15
                self.search_orders_rewarded = True

        elif action_type == "get_order_status" and not self.get_order_status_rewarded:
            status_found = (
                observation.order_status
                and observation.order_status != "Unknown"
            )
            if status_found and task_name in [
                "medium", "ticket_medium", "hard", "ticket_hard"
            ]:
                reward = 0.15
                self.get_order_status_rewarded = True

        elif action_type == "issue_refund":
            if observation.refund_issued and not self.refund_issued:
                self.refund_issued = True
                reward = 0.08

        elif action_type == "reply_and_resolve":
            self.ticket_resolved = True
            can_resolve = False

            if task_name in ["easy", "ticket_easy"]:
                can_resolve = (
                    "password" in msg_lower
                    or "link" in msg_lower
                    or "reset" in msg_lower
                )
            elif task_name in ["medium", "ticket_medium"]:
                can_resolve = (
                    "shipped" in msg_lower
                    or "ord-789" in msg_lower
                    or "order" in msg_lower
                )
            elif task_name in ["hard", "ticket_hard"]:
                can_resolve = self.refund_issued and (
                    "refund" in msg_lower or "ord-111" in msg_lower
                )

            if can_resolve:
                # Give enough reward to reach valid range (0.01, 0.99)
                reward = max(0.0, 0.75 - self.current_reward)

        # Clamp: can never go negative, can never exceed ceiling
        actual_reward = max(0.0, min(reward, 0.99 - self.current_reward))
        self.current_reward += actual_reward
        
        # Update scores with strict clamping - never exactly 0.0 or 1.0
        clamped_score = clamp_score(self.current_reward)
        object.__setattr__(self, "last_score", clamped_score)
        object.__setattr__(self, "score", clamped_score)
        
        return actual_reward
