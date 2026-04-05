from typing import Dict
from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State
from models import TicketSystemAction, TicketSystemObservation

class TicketSystemEnv(EnvClient[TicketSystemAction, TicketSystemObservation, State]):
    """Client for the Ticket System Environment."""
    def _step_payload(self, action: TicketSystemAction) -> Dict:
        return {
            "action_type": action.action_type,
            "customer_id": action.customer_id,
            "order_id": action.order_id,
            "message": action.message,
        }

    def _parse_result(self, payload: Dict) -> StepResult[TicketSystemObservation]:
        obs_data = payload.get("observation", {})
        observation = TicketSystemObservation(
            system_feedback=obs_data.get("system_feedback", ""),
            current_ticket_text=obs_data.get("current_ticket_text", ""),
            ticket_resolved=obs_data.get("ticket_resolved", False),
            orders_found=obs_data.get("orders_found", ""),
            order_status=obs_data.get("order_status", ""),
            refund_issued=obs_data.get("refund_issued", False),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
