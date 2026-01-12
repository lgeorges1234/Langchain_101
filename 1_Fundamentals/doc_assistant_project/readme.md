# DocDacity Intelligent Document Assistant: Project Documentation

## 1. Program Workflow and Lifecycle
The system is built as a modular "State Machine" wrapped in an object-oriented interface. The workflow follows a strict path from initialization to persistent storage.

### Workflow Architecture Schema
The following diagram illustrates the lifecycle of a single user request through the LangGraph engine:



### A. Initialization (Instantiation)
* **Trigger**: The process begins when `main.py` is executed, which instantiates the `DocumentAssistant` class.
* **Setup**: During `__init__`, the assistant prepares the infrastructure:
    * **LLM**: Initializes the `ChatOpenAI` model with specific parameters (temperature 0.1, custom base URL).
    * **Tools**: Calls `get_all_tools()` to generate the "action space" (Search, Reader, Statistics, and Calculator).
    * **Workflow**: Calls `create_workflow()` to build the LangGraph `StateGraph`, connecting nodes and compiling them with an `InMemorySaver` for persistence.

### B. Session Management
* **Start**: `start_session` is called after initialization.
* **Logic**: It either resumes an existing session by reading a JSON file from `./sessions` or generates a new `uuid` for a fresh session. 
* **Persistence**: This `session_id` is mapped to the LangGraph `thread_id`, which acts as the unique key for the checkpointer to save and load conversation history.

### C. The Interaction Loop (`process_message`)
* **Call**: Every user input in the terminal triggers `process_message`.
* **Step 1 (Config)**: It creates a `config` dictionary containing the `thread_id`, the LLM instance, and the tools.
* **Step 2 (Initial State)**: It prepares the `AgentState` by retrieving the `conversation_summary` and `active_documents` from the session storage.
* **Step 3 (Invoke)**: The assistant calls `self.workflow.invoke(initial_state, config=config)`. Control shifts to the Graph.
* **Step 4 (Routing)**:
    1. `classify_intent` uses structured output to decide the `next_step`.
    2. The state moves to a specialized agent (QA, Summarization, or Calculation).
    3. The agent uses the required tools (e.g., `document_reader` or `calculate`).
    4. The state always ends at `update_memory` to compress history and update relevant document IDs.
* **Step 5 (Finalize)**: The results are displayed to the user and the updated session is saved to a JSON file.

## 2. Implementation Decisions

### State and Memory Management
* **Reducer**: The `actions_taken` field in `AgentState` uses `operator.add`. This allows the system to accumulate a list of visited nodes instead of overwriting the value.
* **Checkpointer**: We use `InMemorySaver` to persist state across invocations. This enables "context-aware" follow-up questions where the user can use pronouns like "it" or "that invoice".
* **Summary Compression**: After every turn, the `update_memory` node generates a new `conversation_summary` using the `UpdateMemoryResponse` schema. This ensures the LLM stays within its context window during long conversations.

### Structured Outputs and Validation
* **Pydantic Schemas**: All complex responses are enforced via Pydantic models (e.g., `AnswerResponse`, `UserIntent`). 
* **Constraint Enforcement**: We use Pydantic's validation features, such as `ge=0, le=1` for confidence scores and `Literal` types for `intent_type`, ensuring data integrity.

### Tool Security and Audit
* **Safety**: The `calculate` tool handles mathematical expressions using a `try/except` block to manage errors gracefully.
* **Audit Trail**: Every tool call is logged via the `ToolLogger` class using the `log_tool_use` method. This generates a persistent JSON log in the `./logs` directory for every session.

## 3. Example Conversations

| Interaction Type | User Input | Intent Detected | Expected System Action |
| :--- | :--- | :--- | :--- |
| **Q&A** | "Who is the client for INV-001?" | `qa` | Reads INV-001 and identifies "Acme Corporation". |
| **Calculation** | "Sum INV-001 and INV-002" | `calculation` | Reads both invoices and uses the `calculate` tool for the total. |
| **Follow-up** | "Summarize it" | `summarization` | Uses memory to identify "it" as the previously discussed document. |

## 4. Operational Instructions
1. **Environment**: Place your OpenAI API key in a `.env` file at the project root.
2. **Execution**: Run `python main.py` to start the interactive loop.
3. **Audit**:
    * View conversation snapshots in `./sessions/<session_id>.json`.
    * View tool execution details in `./logs/session_<session_id>.json`.