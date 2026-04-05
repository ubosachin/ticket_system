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

# 🎫 Ticket System Support Environment

A real-world **OpenEnv** environment simulating a professional customer support representative's workflow. This environment allows AI agents to learn how to triage tickets, search customer databases, check order statuses, and resolve complex issues like password resets or defective-item refunds.

---

## 🚀 Getting Started (Zero-Knowledge Guide)

If you are new to OpenEnv, follow these simple steps to get this environment running and testing your first AI agent.

### 1. Local Setup
Ensure you have **Python 3.10+** installed, then run:

```bash
# Clone the repository
git clone https://github.com/ubosachin/ticket_system.git
cd ticket_system

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # Mac/Linux (or venv\Scripts\activate on Windows)

# Install required tools
pip install openenv-core openai pydantic
```

### 2. Configure Your AI "Brain"
The environment needs an AI model to act as the agent. Set your Hugging Face token and model choice:
```bash
export HF_TOKEN="your_huggingface_token"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export MY_ENV_TASK="easy"  # Difficulty: easy, medium, or hard
```

### 3. Run the Evaluation
Launch the automated support agent to see it solve tasks in real-time:
```bash
python inference.py
```

---

## 🛠️ How It Works

### The Workflow
1.  **Initialization**: The environment resets and assigns the agent a specific support ticket (e.g., "I need a refund for my broken laptop").
2.  **Tool Usage**: The agent uses actions like `INFO_SEARCH` to query a mock database of orders (`ORDERS_DB`).
3.  **Grading**: A programmatic **Rubric** evaluates every step. The agent earns partial rewards for intermediate successes (like finding a tracking ID) and a full reward (1.0) for resolving the ticket correctly.

### Difficulty Levels
| Level | Task | Objective |
| :--- | :--- | :--- |
| **Easy** | Password Reset | Provide instructions for a standard password reset request. |
| **Medium** | Order Status | Correct customers' tracking queries using database lookups. |
| **Hard** | Damaged Goods | Verify the order, process a refund, and reply to the customer. |

---

## 📊 Action & Observation Space

### Actions (`TicketSystemAction`)
- `action_type`: `INFO_SEARCH`, `REFUND_PROCESS`, or `MESSAGE_CUSTOMER`.
- `customer_id` / `order_id`: Database query parameters.
- `message`: The agent's final text response to the user.

### Observations (`TicketSystemObservation`)
- `system_feedback`: Results from database queries or system actions.
- `current_ticket_text`: The customer's original complaint.
- `reward`: The current performance score (0.0 to 1.0).
- `done`: Set to `True` when the ticket is successfully resolved.

---

## 🌐 Online Access
You can test this environment manually through the web interface on Hugging Face:
**[https://huggingface.co/spaces/ubosachin/ticket_system](https://huggingface.co/spaces/ubosachin/ticket_system)**

---

*Built for the Meta OpenEnv Hackathon 2026.*
