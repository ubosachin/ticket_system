"""
Direct test of grader validation to ensure scores are in correct range.
"""
import os
import sys
from server.ticket_system_environment import TicketSystemEnvironment
from models import TicketSystemAction

def test_task(task_id: str):
    """Test a single task and return the final score."""
    print(f"\n{'='*60}")
    print(f"Testing task: {task_id}")
    print(f"{'='*60}")
    
    env = TicketSystemEnvironment()
    obs = env.reset(task=task_id)
    
    print(f"Initial reward (from reset): {obs.reward}")
    print(f"Rubric current_reward: {env.rubric.current_reward}")
    print(f"Rubric last_score: {getattr(env.rubric, 'last_score', 'NOT SET')}")
    
    # Get optimal actions for this task
    if task_id in ["easy", "ticket_easy"]:
        actions = [
            {"action_type": "read_ticket"},
            {
                "action_type": "reply_and_resolve",
                "message": "Hi! I have reset your password. Please follow this reset link to create a new password.",
            },
        ]
    elif task_id in ["medium", "ticket_medium"]:
        actions = [
            {"action_type": "read_ticket"},
            {"action_type": "search_orders", "customer_id": "CUST-456"},
            {"action_type": "get_order_status", "order_id": "ORD-789"},
            {
                "action_type": "reply_and_resolve",
                "message": "Your order ORD-789 has been shipped and is on its way. You will receive it soon.",
            },
        ]
    else:  # hard
        actions = [
            {"action_type": "read_ticket"},
            {"action_type": "search_orders", "customer_id": "CUST-999"},
            {"action_type": "get_order_status", "order_id": "ORD-111"},
            {"action_type": "issue_refund", "order_id": "ORD-111"},
            {
                "action_type": "reply_and_resolve",
                "message": "I'm sorry about the damaged item. I have issued a full refund for order ORD-111. Please allow 3-5 business days for it to appear.",
            },
        ]
    
    # Execute actions
    for i, action_kwargs in enumerate(actions, 1):
        action = TicketSystemAction(**action_kwargs)
        obs = env.step(action)
        print(f"  Step {i}: reward={obs.reward:.4f}, cumulative_rubric={env.rubric.current_reward:.4f}")
    
    final_score = obs.reward
    final_cumulative = env.rubric.current_reward
    final_last_score = getattr(env.rubric, 'last_score', None)
    
    print(f"\nFinal per-step reward (from observation): {final_score:.4f}")
    print(f"Final cumulative reward (rubric.current_reward): {final_cumulative:.4f}")
    print(f"Final last_score (rubric.last_score): {final_last_score}")
    print(f"Rubric last_score: {getattr(env.rubric, 'last_score', 'NOT SET')}")
    
    # The platform likely uses the CUMULATIVE score
    score_to_validate = final_cumulative
    valid = 0 < score_to_validate < 1
    print(f"✅ Cumulative score valid (0 < {score_to_validate:.4f} < 1): {valid}")
    
    if not valid:
        print(f"❌ INVALID: Score {score_to_validate:.4f} is NOT strictly between 0 and 1")
    
    return score_to_validate, valid

if __name__ == "__main__":
    print("Testing all 3 tasks with direct validation\n")
    
    scores = {}
    all_valid = True
    
    for task in ["ticket_easy", "ticket_medium", "ticket_hard"]:
        score, valid = test_task(task)
        scores[task] = (score, valid)
        all_valid = all_valid and valid
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for task, (score, valid) in scores.items():
        status = "✅" if valid else "❌"
        print(f"{status} {task}: {score:.4f}")
    
    print(f"\nAll valid: {all_valid}")
    sys.exit(0 if all_valid else 1)
