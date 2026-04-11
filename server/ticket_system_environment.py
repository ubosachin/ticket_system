import os
from typing import Any, Optional
from uuid import uuid4
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    # First try absolute imports (standard when running from root)
    from models import TicketSystemAction, TicketSystemObservation
except (ImportError, ModuleNotFoundError):
    # Fallback to relative imports if needed
    from ..models import TicketSystemAction, TicketSystemObservation

# Database for the environment
ORDERS_DB = {
    "CUST-456": {"order_id": "ORD-789", "status": "Shipped", "item": "Laptop"},
    "CUST-999": {"order_id": "ORD-111", "status": "Delivered", "item": "Vase"}
}

from .rubric import TicketSystemRubric

class TicketSystemEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self.rubric = TicketSystemRubric()
        super().__init__(rubric=self.rubric)
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.task_name = os.getenv("MY_ENV_TASK", "easy")
        self.reset_env()

    def reset_env(self):
        self._state.step_count = 0
        self.current_ticket_text = ""
        self.ticket_resolved = False
        self.refund_issued = False
        self.orders_found = ""
        self.order_status = ""
        self.system_feedback = "Welcome to the Ticket Support System."
        self.max_reward = 0.8
        self.current_reward = 0.2

        if self.task_name in ["easy", "ticket_easy"]:
            self.current_ticket_text = "I forgot my password, my username is CUST-123. Can you help?"
        elif self.task_name in ["medium", "ticket_medium"]:
            self.current_ticket_text = "Where is my order? I am CUST-456."
        elif self.task_name in ["hard", "ticket_hard"]:
            self.current_ticket_text = "I received a broken item for my recent order. I am CUST-999. I want a refund."
        else:
            # Default fallback
            self.task_name = "easy"
            self.current_ticket_text = "I forgot my password, my username is CUST-123. Can you help?"

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> TicketSystemObservation:
        """
        Reset the environment.

        Returns:
            TicketSystemObservation with a ready message
        """
        # Handle task selection from kwargs (common in OpenEnv)
        if "task" in kwargs:
            self.task_name = kwargs["task"]
        elif "task_id" in kwargs:
            self.task_name = kwargs["task_id"]

        self._state = State(episode_id=episode_id or str(uuid4()), step_count=0)
        self.reset_env()
        self._reset_rubric()
        # Ensure the rubric matches the environment's initial reward
        self.rubric.current_reward = self.current_reward
        return self._make_obs(reward=self.current_reward)

    def _make_obs(self, reward=None, done=False):
        if reward is None:
            reward = self.current_reward
        return TicketSystemObservation(
            system_feedback=self.system_feedback,
            current_ticket_text=self.current_ticket_text,
            ticket_resolved=self.ticket_resolved,
            orders_found=self.orders_found,
            order_status=self.order_status,
            refund_issued=self.refund_issued,
            reward=reward,
            done=done,
            task=self.task_name,
            step=self._state.step_count,
            metadata={"step": self._state.step_count, "task": self.task_name}
        )

    def step(
        self,
        action: TicketSystemAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> TicketSystemObservation:
        """
        Execute a step in the environment.

        Args:
            action: TicketSystemAction containing the support agent action
            timeout_s: Optional timeout for the step

        Returns:
            TicketSystemObservation with the results of the action
        """
        self._state.step_count += 1
        reward = 0.0
        done = False

        if self.ticket_resolved:
            self.system_feedback = "Ticket is already resolved."
            return self._make_obs(reward=0.0, done=True)

        if action.action_type == "read_ticket":
            self.system_feedback = "Ticket read."
            reward = 0.05 # minor reward for reading
            
        elif action.action_type == "search_orders":
            if action.customer_id in ORDERS_DB:
                self.orders_found = ORDERS_DB[action.customer_id]["order_id"]
                self.system_feedback = f"Found order: {self.orders_found}"
                if self.task_name in ["medium", "ticket_medium", "hard", "ticket_hard"]:
                    reward = 0.2 # partial progress
            else:
                self.orders_found = "No orders found."
                self.system_feedback = self.orders_found
                
        elif action.action_type == "get_order_status":
            found = False
            for cust, details in ORDERS_DB.items():
                if details["order_id"] == action.order_id:
                    self.order_status = details["status"]
                    self.system_feedback = f"Order status is {self.order_status}"
                    found = True
                    if self.task_name in ["medium", "ticket_medium", "hard", "ticket_hard"]:
                        reward = 0.2 # partial progress
            if not found:
                self.order_status = "Unknown"
                self.system_feedback = "Order ID not found."
                
        elif action.action_type == "issue_refund":
            if action.order_id == "ORD-111" and self.task_name in ["hard", "ticket_hard"]:
                self.refund_issued = True
                self.system_feedback = "Refund issued for ORD-111."
                reward = 0.1 # partial progress
            else:
                self.system_feedback = "Cannot issue refund. Invalid Order ID or not permitted."
                
        elif action.action_type == "reply_and_resolve":
            self.ticket_resolved = True
            done = True
            self.system_feedback = "Ticket resolved and closed."
            
        else:
            self.system_feedback = f"Invalid action_type: {action.action_type}"

        # Grading logic using the rubric
        reward = self._apply_rubric(action, self._make_obs(reward=0.0, done=done))
        # Environment's current_reward tracks total for state consistency
        self.current_reward += reward


        # CRITICAL: Update last_score to cumulative reward AFTER rubric forward
        # The rubric's parent class overwrites last_score, so we set it here
        object.__setattr__(self.rubric, "last_score", self.current_reward)

        if self._state.step_count >= 10:
            done = True
            self.system_feedback = "Max steps reached."

        # Return observation with CUMULATIVE reward (not just per-step)
        return self._make_obs(reward=self.current_reward, done=done)

    @property
    def state(self) -> State:
        return self._state
    
    def get_grader(self):
        """Explicitly expose rubric/grader."""
        return self.rubric
    
    def get_task_scores(self):
        """Get all task scores."""
        return {
            "task": self.task_name,
            "score": self.rubric.current_reward,
            "last_score": self.rubric.last_score,
        }
