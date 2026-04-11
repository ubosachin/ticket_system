"""
Diagnostic script to verify what the platform would see.
Tests direct environment instantiation like the validator does.
"""
import sys
import traceback
from server.ticket_system_environment import TicketSystemEnvironment
from models import TicketSystemAction

def diagnose_task(task_id: str):
    """Directly test task instantiation and scoring."""
    print(f"\n{'='*70}")
    print(f"DIAGNOSING TASK: {task_id}")
    print(f"{'='*70}")
    
    try:
        # Step 1: Create environment
        print(f"✓ Creating environment...")
        env = TicketSystemEnvironment()
        
        # Step 2: Check rubric exists
        print(f"✓ Rubric exists: {env.rubric is not None}")
        print(f"  - Rubric type: {type(env.rubric).__name__}")
        
        # Step 3: Reset with task
        print(f"✓ Resetting with task={task_id}...")
        try:
            obs = env.reset(task=task_id)
            print(f"✓ Reset successful")
            print(f"  - Initial reward: {obs.reward}")
            print(f"  - Task name: {obs.task}")
            print(f"  - Rubric current_reward: {env.rubric.current_reward}")
            print(f"  - Rubric last_score: {getattr(env.rubric, 'last_score', 'NOT SET')}")
        except Exception as e:
            print(f"✗ Reset FAILED: {e}")
            traceback.print_exc()
            return False, 0.0
        
        # Step 4: Get optimal actions
        if task_id in ["easy", "ticket_easy"]:
            actions = [
                {"action_type": "read_ticket"},
                {"action_type": "reply_and_resolve", "message": "Hi! I have reset your password. Please follow this reset link to create a new password."},
            ]
        elif task_id in ["medium", "ticket_medium"]:
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
        
        print(f"✓ Executing {len(actions)} actions...")
        for i, action_data in enumerate(actions, 1):
            try:
                action = TicketSystemAction(**action_data)
                obs = env.step(action)
                print(f"  Step {i}: action_type={action_data['action_type']:<20} reward={obs.reward:.4f} rubric={env.rubric.current_reward:.4f}")
            except Exception as e:
                print(f"  ✗ Step {i} FAILED: {e}")
                traceback.print_exc()
                return False, 0.0
        
        # Step 5: Get final score
        final_score = obs.reward if hasattr(obs, 'reward') else 0.0
        final_cumulative = env.rubric.current_reward
        final_last_score = getattr(env.rubric, 'last_score', None)
        
        print(f"\n✓ FINAL SCORES for {task_id}:")
        print(f"  - Observation reward: {final_score:.4f}")
        print(f"  - Rubric cumulative: {final_cumulative:.4f}")
        print(f"  - Rubric last_score: {final_last_score}")
        
        # Check validity
        valid = 0 < final_cumulative < 1
        print(f"  - Valid (0 < {final_cumulative:.4f} < 1): {valid}")
        
        if not valid:
            print(f"  ✗ INVALID SCORE: {final_cumulative}")
            return False, final_cumulative
        
        env.close()
        return True, final_cumulative
        
    except Exception as e:
        print(f"✗ FATAL ERROR for {task_id}: {e}")
        traceback.print_exc()
        return False, 0.0

if __name__ == "__main__":
    print("DIAGNOSTIC TEST - Simulating Platform Validator")
    print("="*70)
    
    results = {}
    all_passed = True
    
    for task in ["ticket_easy", "ticket_medium", "ticket_hard"]:
        passed, score = diagnose_task(task)
        results[task] = (passed, score)
        all_passed = all_passed and passed
    
    print(f"\n\n{'='*70}")
    print("DIAGNOSTIC SUMMARY")
    print(f"{'='*70}")
    
    for task, (passed, score) in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        valid = "valid" if (0 < score < 1) else "INVALID"
        print(f"{status} {task:20} score={score:.4f} ({valid})")
    
    print(f"\nAll tasks passed: {all_passed}")
    print(f"All scores valid (0 < score < 1): {all(0 < s < 1 for _, s in results.values())}")
    
    sys.exit(0 if all_passed and all(0 < s < 1 for _, s in results.values()) else 1)
