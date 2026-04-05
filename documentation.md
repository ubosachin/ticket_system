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
  - `0.2` for searching the correct customer ID.
  - `0.3` for correctly identifying order status.
  - `0.5` for successfully resolving the ticket with the correct final answer.
  - `-0.1` penalty for invalid actions or malformed JSON.

### C. Models (`models.py`)
- **TicketSystemAction**: Defines the schema for agent clicks and messages.
  - `action_type`: `INFO_SEARCH` (search DB), `REFUND_PROCESS` (issue refund), `MESSAGE_CUSTOMER` (reply and resolve).
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

*This environment is designed for the Meta OpenEnv hackathon, focusing on robust tool usage and multi-step reasoning capabilities.*
