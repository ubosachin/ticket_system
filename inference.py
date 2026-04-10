import asyncio
import os
import textwrap
import json
from typing import List, Optional

from openai import OpenAI
from pydantic import ValidationError

from models import TicketSystemAction
from client import TicketSystemEnv
from server.ticket_system_environment import TicketSystemEnvironment

LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
API_KEY = os.getenv("HF_TOKEN")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
TASK_NAME = os.getenv("MY_ENV_TASK", "easy")
BENCHMARK = "ticket_system"
MAX_STEPS = 8
TEMPERATURE = 0.7
MAX_TOKENS = 150
SUCCESS_SCORE_THRESHOLD = 0.1

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an AI Support Agent interacting with a Ticket Support System.
    You must resolve the customer's query.
    Return your actions as a JSON object adhering to this schema:
    {"action_type": "<action>", "customer_id": "<id>", "order_id": "<id>", "message": "<msg>"}
    
    Allowed action_types: read_ticket, search_orders, get_order_status, issue_refund, reply_and_resolve.
    - search_orders requires 'customer_id'
    - get_order_status and issue_refund require 'order_id'
    - reply_and_resolve requires 'message'
    
    Do not reply with any markdown formatting or text outside the JSON object.
    Just raw JSON.
    """
).strip()

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

def build_user_prompt(step: int, obs, last_reward: float, history: List[str]) -> str:
    history_block = "\\n".join(history[-4:]) if history else "None"
    return textwrap.dedent(
        f"""
        Step: {step}
        Observation space:
        - system_feedback: {obs.system_feedback}
        - current_ticket_text: {obs.current_ticket_text}
        - ticket_resolved: {obs.ticket_resolved}
        - orders_found: {obs.orders_found}
        - order_status: {obs.order_status}
        - refund_issued: {obs.refund_issued}
        
        Last reward: {last_reward:.2f}
        Previous steps:
        {history_block}
        
        Send your next JSON action.
        """
    ).strip()

def get_model_message(client: OpenAI, step: int, obs, last_reward: float, history: List[str]) -> str:
    user_prompt = build_user_prompt(step, obs, last_reward, history)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        # Some simple clean up in case model adds quotes
        if text.startswith("```json"): text = text[7:]
        if text.startswith("```"): text = text[3:]
        if text.endswith("```"): text = text[:-3]
        return text.strip()
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return '{"action_type": "read_ticket"}'

async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # Standard pattern: connect via client if image exists, else fallback to class
    try:
        if LOCAL_IMAGE_NAME:
            # Connect to a Docker container (standard for evaluation)
            env = await TicketSystemEnv.from_docker_image(LOCAL_IMAGE_NAME, env_vars={"MY_ENV_TASK": TASK_NAME})
        else:
            # Fallback for local dev testing
            env = TicketSystemEnvironment()
    except Exception as e:
        print(f"[DEBUG] Falling back to local env due to: {e}", flush=True)
        env = TicketSystemEnvironment()

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        if getattr(env, "reset_async", None):
            res_reset = await env.reset_async(task=TASK_NAME)
        elif asyncio.iscoroutinefunction(getattr(env, "reset", None)):
             res_reset = await env.reset(task=TASK_NAME)
        else:
            res_reset = env.reset(task=TASK_NAME)
            
        obs = res_reset if not hasattr(res_reset, "observation") else res_reset.observation
        done = res_reset.done if hasattr(res_reset, "done") else False
            
        last_reward = res_reset.reward if hasattr(res_reset, "reward") else 0.1
        rewards.append(last_reward)

        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            message = get_model_message(client, step, obs, last_reward, history)
            steps_taken = step
            error = None

            try:
                action_data = json.loads(message)
                action = TicketSystemAction(**action_data)
                
                if getattr(env, "step_async", None):
                    res_step = await env.step_async(action)
                elif asyncio.iscoroutinefunction(getattr(env, "step", None)):
                    res_step = await env.step(action)
                else:
                    res_step = env.step(action)
                    
                obs = res_step if getattr(res_step, "system_feedback", None) else res_step.observation
                reward = obs.reward if hasattr(obs, "reward") else 0.0
                done = obs.done if hasattr(obs, "done") else False
                
            except json.JSONDecodeError as e:
                error = f"JSON parse error: {e}"
                reward = -0.1
                done = False
            except ValidationError as e:
                error = f"Action validation error: {e}"
                reward = -0.1
                done = False
            except Exception as e:
                error = str(e)
                reward = -0.1
                done = False

            rewards.append(reward)
            last_reward = reward
            
            # compress JSON for logging
            compressed_action = "".join(message.split())
            log_step(step=step, action=compressed_action, reward=reward, done=done, error=error)
            history.append(f"Step {step}: {compressed_action} -> reward {reward:+.2f}")

            if done:
                break

        score = sum(rewards)
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        try:
            if hasattr(env, "close"):
                env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())
