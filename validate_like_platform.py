"""
Simulate EXACTLY what the platform validator does.
Direct environment instantiation, no inference.py.
"""
import sys
from server.ticket_system_environment import TicketSystemEnvironment
from models import TicketSystemAction

def validate_task_directly(task_id: str):
    """
    Direct validation - what the platform likely does.
    """
    print(f"\n{'='*70}")
    print(f"VALIDATOR CHECK: {task_id}")
    print(f"{'='*70}")
    
    # Step 1: Create environment
    env = TicketSystemEnvironment()
    
    # Step 2: Reset with task
    obs = env.reset(task=task_id)
    print(f"Reset: reward={obs.reward}, rubric.current_reward={env.rubric.current_reward}")
    print(f"       rubric.last_score={env.rubric.last_score}")
    print(f"       rubric.score={getattr(env.rubric, 'score', 'NO PROPERTY')}")
    
    # Step 3: Get actions
    if task_id == "ticket_easy":
        actions = [
            {"action_type": "read_ticket"},
            {"action_type": "reply_and_resolve", "message": "Hi! I have reset your password. Please follow this reset link to create a new password."},
        ]
    elif task_id == "ticket_medium":
        actions = [
            {"action_type": "read_ticket"},
            {"action_type": "search_orders", "customer_id": "CUST-456"},
            {"action_type": "get_order_status", "order_id": "ORD-789"},
            {"action_type": "reply_and_resolve", "message": "Your order ORD-789 has been shipped and is on its way. You will receive it soon."},
        ]
    else:
        actions = [
            {"action_type": "read_ticket"},
            {"action_type": "search_orders", "customer_id": "CUST-999"},
            {"action_type": "get_order_status", "order_id": "ORD-111"},
            {"action_type": "issue_refund", "order_id": "ORD-111"},
            {"action_type": "reply_and_resolve", "message": "I'm sorry about the damaged item. I have issued a full refund for order ORD-111. Please allow 3-5 business days for it to appear."},
        ]
    
    # Step 4: Execute
    for action_data in actions:
        action = TicketSystemAction(**action_data)
        obs = env.step(action)
    
    # Step 5: Get FINAL score - what validator checks
    final_obs_reward = obs.reward
    final_rubric_current = env.rubric.current_reward
    final_rubric_last_score = env.rubric.last_score
    final_rubric_score = getattr(env.rubric, 'score', None)
    
    print(f"\nFinal state:")
    print(f"  obs.reward:          {final_obs_reward:.4f}")
    print(f"  rubric.current_reward: {final_rubric_current:.4f}")
    print(f"  rubric.last_score:   {final_rubric_last_score:.4f}")
    print(f"  rubric.score:        {final_rubric_score}")
    
    # Check validity - the validator likely checks one of these
    candidates = [
        ("obs.reward", final_obs_reward),
        ("rubric.current_reward", final_rubric_current),
        ("rubric.last_score", final_rubric_last_score),
        ("rubric.score", final_rubric_score),
    ]
    
    print(f"\nValidity check (0 < score < 1):")
    all_valid = True
    for name, score in candidates:
        if score is not None:
            valid = 0 < score < 1
            status = "✅" if valid else "❌"
            print(f"  {status} {name:25} = {score:.4f} (valid={valid})")
            if not valid:
                all_valid = False
    
    env.close()
    return all_valid

if __name__ == "__main__":
    print("PLATFORM VALIDATOR SIMULATION")
    print("="*70)
    
    all_passed = True
    for task in ["ticket_easy", "ticket_medium", "ticket_hard"]:
        passed = validate_task_directly(task)
        if not passed:
            all_passed = False
    
    print(f"\n\n{'='*70}")
    print(f"RESULT: {'✅ ALL VALID' if all_passed else '❌ SOME INVALID'}")
    print(f"{'='*70}")
    
    sys.exit(0 if all_passed else 1)
