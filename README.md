---
title: Ticket System Environment
emoji: 🎫
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
tags:
  - openenv
---

# Ticket System Support Environment

A real-world OpenEnv environment simulating a customer support representative's workflow. This environment allows AI agents to learn how to triage tickets, search for customer information, check order statuses, and resolve issues like password resets or refunds.

## Environment Overview

In this environment, an agent takes on the role of a support representative. The environment provides a series of tasks with increasing difficulty. Success is measured by how accurately and efficiently the agent resolves the customer's request.

### Motivation
Customer support is a high-impact real-world task for AI agents. This environment provides a controlled but realistic space to evaluate an agent's ability to:
1. **Understand** natural language support tickets.
2. **Interact** with simulated internal tools (databases).
3. **Execute** multi-step logic (e.g., verify order -> check status -> issue refund).
4. **Communicate** effectively with the customer to resolve the issue.

## Action Space

The agent interacts using `TicketSystemAction` which includes:
- `action_type`: One of `read_ticket`, `search_orders`, `get_order_status`, `issue_refund`, `reply_and_resolve`.
- `customer_id`: Used with `search_orders`.
- `order_id`: Used with `get_order_status` or `issue_refund`.
- `message`: Used with `reply_and_resolve` to communicate with the customer.

## Observation Space

The `TicketSystemObservation` provides:
- `system_feedback`: Result of the last action (e.g., "Found order: ORD-789").
- `current_ticket_text`: The text of the active support ticket.
- `ticket_resolved`: Boolean indicating if the ticket is closed.
- `orders_found`: List of orders associated with a customer.
- `order_status`: Status of a specific order (e.g., "Shipped", "Delivered").
- `refund_issued`: Boolean indicating if a refund was processed.

## Tasks

| Task ID | Difficulty | Objective |
|---------|------------|-----------|
| **easy** | Easy | Resolve a simple password reset query. |
| **medium** | Medium | Find a customer's order and report its shipping status. |
| **hard** | Hard | Process a refund for a broken item after verifying the order. |

## Reward Function

The environment uses a shaped reward function (0.0 to 1.0) to provide feedback on partial progress:
- **0.05**: For reading the ticket.
- **0.20**: For correctly searching for customer orders (Medium/Hard).
- **0.20**: For correctly checking order status (Medium/Hard).
- **0.30**: For successfully issuing a refund (Hard).
- **Final**: The remaining reward (up to 1.0) is granted when the ticket is successfully resolved with the correct information.

## Setup and Usage

### Prerequisites
- Docker
- Python 3.9+
- `openenv-core` (`pip install openenv-core`)

### Build the Environment
```bash
docker build -t ticket_system:latest -f server/Dockerfile .
```

### Run Locally
```bash
docker run -p 8000:8000 ticket_system:latest
```

### Run Inference Baseline
To run the baseline agent against the environment:
```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="your-hf-token"
export MY_ENV_TASK="easy"
python3 inference.py
```

## Baseline Scores

| Task | Score (Agent) | Success Rate |
|------|---------------|--------------|
| Easy | 1.00 | 100% |
| Medium | 1.00 | 100% |
| Hard | 1.00 | 100% |

*Scores based on Qwen 2.5 72B Instruct baseline.*

## OpenEnv Compliance
This environment fully implements the OpenEnv specification:
- ✅ Typed Pydantic models for Actions/Observations.
- ✅ Full `step()`, `reset()`, `state()` API.
- ✅ `openenv.yaml` manifest.
- ✅ Programmatic graders via internal reward logic.
