import enum
from openenv.core.env_server.types import Action, Observation
from pydantic import Field
from typing import Optional

class TicketSystemAction(Action):
    """Action for the Ticket System Support environment."""
    action_type: str = Field(..., description="One of: 'read_ticket', 'search_orders', 'get_order_status', 'issue_refund', 'reply_and_resolve'")
    customer_id: str = Field(default="", description="Customer ID to search orders for. Use with 'search_orders'")
    order_id: str = Field(default="", description="Order ID to get status or issue refund. Use with 'get_order_status' or 'issue_refund'")
    message: str = Field(default="", description="Message to reply to the customer. Use with 'reply_and_resolve'")

class TicketSystemObservation(Observation):
    """Observation from the Ticket System Support environment."""
    system_feedback: str = Field(default="", description="Feedback from the last action")
    current_ticket_text: str = Field(default="", description="The text of the active support ticket")
    ticket_resolved: bool = Field(default=False, description="Whether the ticket is resolved")
    orders_found: str = Field(default="", description="JSON string of orders found")
    order_status: str = Field(default="", description="Status of the queried order")
    refund_issued: bool = Field(default=False, description="Whether a refund was successfully issued")
    task: str = Field(default="easy", description="The name of the current task")
    step: int = Field(default=0, description="The current step number")
