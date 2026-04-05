from typing import Any
from openenv.core.rubrics import Rubric

class TicketSystemRubric(Rubric):
    """Rubric for the Ticket System environment to compute rewards and scores."""
    
    def __init__(self):
        super().__init__()
        self.reset()
        
    def reset(self):
        self.current_reward = 0.0
        self.refund_issued = False
        self.ticket_resolved = False

    def forward(self, action: Any, observation: Any) -> float:
        # Grader logic matching the environment task configuration
        task_name = observation.metadata.get("task", "easy")
        action_type = action.action_type
        msg_lower = action.message.lower()
        
        reward = 0.0
        
        if self.ticket_resolved:
            return 0.0

        if action_type == "read_ticket":
            reward = 0.05
            
        elif action_type == "search_orders":
            if observation.orders_found and observation.orders_found != "No orders found.":
                if task_name in ["medium", "hard"]:
                    reward = 0.2
                    
        elif action_type == "get_order_status":
            if observation.order_status and observation.order_status != "Unknown":
                if task_name in ["medium", "hard"]:
                    reward = 0.2
                    
        elif action_type == "issue_refund":
            if observation.refund_issued:
                self.refund_issued = True
                reward = 0.3
                
        elif action_type == "reply_and_resolve":
            self.ticket_resolved = True
            if task_name == "easy":
                if "password" in msg_lower or "link" in msg_lower or "reset" in msg_lower:
                    reward = 1.0 - self.current_reward
            elif task_name == "medium":
                if "shipped" in msg_lower or "ord-789" in msg_lower:
                    reward = 1.0 - self.current_reward
            elif task_name == "hard":
                if self.refund_issued and ("refund" in msg_lower or "ord-111" in msg_lower):
                    reward = 1.0 - self.current_reward
        
        # Maximize and clamp reward
        actual_reward = max(0.0, min(reward, 1.0 - self.current_reward))
        self.current_reward += actual_reward
        return actual_reward
