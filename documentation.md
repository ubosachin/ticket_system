# Ticket System OpenEnv Documentation

Welcome to the **Ticket System OpenEnv**, a real-world task environment built to evaluate AI agent capabilities in customer support and data triage. This document explains the architecture, task levels, grading logic, and how the environment operates.

---

## 1. Overview
The **Ticket System OpenEnv** simulates a professional customer support workflow where an AI agent acts as a Support Assistant. The agent must interact with a mock ticketing system to resolve customer queries by performing actions such as searching for order records, checking statuses, and processing refunds.

---

## 2. Environment Structure
The environment is built using the **OpenEnv Core** framework and consists of several key components:

### A. Core Environment (`server/ticket_system_environment.py`)
- **State Machine**: Manages the current ticket being handled, the step count, and the simulation state.
- **Task Switching**: Supports three distinct difficulty levels through the `MY_ENV_TASK` variable (`easy`, `medium`, `hard`).
- **Internal Database**: Contains mock customer and order data to simulate "real-world" database lookups.

### B. Programmatic Rubric (`server/rubric.py`)
- **Grading Logic**: Evaluates agent performance after every step.
- **Reward Signals**:
  - `0.01` baseline reward for starting the task (ensures non-zero score).
  - `0.05` for reading the ticket.
  - `0.20` for searching for orders.
  - `0.20` for getting order status.
  - `0.30` for issuing a refund (Hard task).
  - `0.99` total max score for successfully resolving the ticket (strictly below 1.0).
  - `-0.1` penalty for invalid actions or malformed JSON.

### C. Models (`models.py`)
- **TicketSystemAction**: Defines the schema for agent clicks and messages.
  - `action_type`: `read_ticket`, `search_orders`, `get_order_status`, `issue_refund`, `reply_and_resolve`.
  - `customer_id` / `order_id`: Query parameters for database lookups.
  - `message`: Final response to the customer.
- **TicketSystemObservation**: What the agent "sees" back from the system (system feedback, order status, ticket text).

---

## 3. Tasks & Difficulty Levels

| Level | Task Description | Goal |
| :--- | :--- | :--- |
| **Easy** | **Password Reset Query** | The agent must identify that the user is asking for a password reset and provide instructions. |
| **Medium** | **Order Status Lookup** | The agent must search the mock database using the provided `customer_id` to find the status of an order. |
| **Hard** | **Refund Processing** | The agent must verify the order status first, see that it's defective, and process a refund before closing the ticket. |

---

## 4. How it Works (The Step-by-Step Cycle)

### Step 1: Initialization (`reset()`)
- The environment is initialized with a specific task based on the `task` parameter or `MY_ENV_TASK` variable.
- A new ticket is "assigned" to the agent, described in the `current_ticket_text`.

### Step 2: Agent Inference
- The agent (running via `inference.py`) reads the ticket.
- It determines the necessary action (e.g., "I need to check the database for customer `CUST-456`").

### Step 3: Action Execution (`step()`)
- The agent sends a `TicketSystemAction` JSON to the environment.
- The environment executes the logic:
  - If `INFO_SEARCH`: It queries the mock database and returns the result (e.g., "Order ORD-789 is Shipped").
  - If `REFUND_PROCESS`: It checks if a refund is valid and updates the internal state.
  - If `MESSAGE_CUSTOMER`: It checks if the message resolves the user's intent.

### Step 4: Feedback & Reward
- The agent receives a `TicketSystemObservation` containing:
  - `system_feedback`: A log of what happened (e.g., "Database query successful").
  - `reward`: A partial progress signal (0.0 to 1.0).
  - `done`: Set to `True` if the ticket is resolved or max steps are reached.

---

## 5. Baseline Evaluation (`inference.py`)
The `inference.py` script serves as the baseline auditor. It:
1.  **Starts the Environment**: Connects to the server or Docker container.
2.  **Prompts the LLM**: Uses a system prompt to guide the AI on how to use the available tools.
3.  **Logs Interactions**: Records the session in a structured format (`[START]`, `[STEP]`, `[END]`).
4.  **Calculates Final Score**: The final score is the maximum reward achieved during the episode.

---

## 6. Deployment & Compliance
- **Dockerized**: The `server/` directory contains a production-ready `Dockerfile`.
- **Hugging Face Ready**: Fully compatible with HF Spaces, including a web interface for manual testing.
- **Typing**: Strictly typed using Pydantic for both actions and observations.

---

## 7. Getting Started (For Beginners)

If you have zero knowledge of OpenEnv, follow these simple steps to get this environment running and testing your first AI agent.

### A. Local Setup (Installing the Tools)
1.  **Install Python**: Ensure you have Python 3.10+ installed.
2.  **Clone the Repo**:
    ```bash
    git clone https://github.com/ubosachin/ticket_system.git
    cd ticket_system
    ```
3.  **Create a Virtual Environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # Mac/Linux
    # OR: venv\Scripts\activate (Windows)
    ```
4.  **Install Dependencies**:
    ```bash
    pip install openenv-core openai pydantic
    ```

### B. Configuring Your Credentials
The environment needs to know how to talk to an AI model (like GPT-4 or Qwen). Set these variables in your terminal:
```bash
export HF_TOKEN="your_huggingface_write_token"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export MY_ENV_TASK="easy"  # Change to "medium" or "hard" as you progress
```

### C. Running Your First Test (Baseline Audit)
The `inference.py` script is the main "brain" that runs the simulation. To start the agent:
```bash
python inference.py
```
**What happens when you run this?**
- It creates a "Support Assistant" agent.
- It resets the ticket system and gives the agent a new customer problem.
- It loops automatically, performing actions and giving you live progress logs.

### D. Using the Hugging Face Web Interface
You can also test the environment **without writing code**!
1.  Go to: **[https://huggingface.co/spaces/ubosachin/ticket_system](https://huggingface.co/spaces/ubosachin/ticket_system)**
2.  On the **App** tab, you will see a UI.
3.  You can manually enter **Action Type**, **Customer ID**, and **Message** to see how the system responds.

---

## 8. Summary of Use
- **Goal**: Help the customer solve their issue.
- **Workflow**: `Reset (Get Ticket)` -> `Step (Search DB)` -> `Step (Reply/Process)` -> `Done`.
- **Measurement**: Your success is measured by the `reward` (strictly between 0.0 and 1.0) shown in the logs.

*Happy Coding! For any issues, refer to the [OpenEnv Spec](https://github.com/meta-pytorch/openenv).*
