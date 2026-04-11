"""
Test last_score directly
"""
from server.ticket_system_environment import TicketSystemEnvironment
from models import TicketSystemAction

env = TicketSystemEnvironment()
obs = env.reset(task="ticket_easy")

print(f"After reset:")
print(f"  current_reward: {env.rubric.current_reward}")
print(f"  last_score: {env.rubric.last_score}")
print(f"  __dict__: {env.rubric.__dict__}")
print()

# Step 1
action = TicketSystemAction(action_type="read_ticket")
obs = env.step(action)
print(f"After step 1 (read_ticket):")
print(f"  current_reward: {env.rubric.current_reward}")
print(f"  last_score: {env.rubric.last_score}")
print(f"  returned reward: {obs.reward}")
print()

# Step 2
action = TicketSystemAction(
    action_type="reply_and_resolve",
    message="Hi! I have reset your password. Please follow this reset link to create a new password."
)
obs = env.step(action)
print(f"After step 2 (reply_and_resolve):")
print(f"  current_reward: {env.rubric.current_reward}")
print(f"  last_score: {env.rubric.last_score}")
print(f"  returned reward: {obs.reward}")
print(f"  __dict__: {env.rubric.__dict__}")
