# Plan for `llm-map` Crate (Test-Focused)

This plan outlines the steps to create the `llm-map` Rust subcrate, emphasizing a test-driven approach.

**Phase 1: Project Setup & Structure**

1.  **Create Directories:**
    *   `crates/llm-map`
    *   `crates/llm-map/src`
    *   `crates/llm-map/tests` (For integration tests)
2.  **Create Core Files:**
    *   `crates/llm-map/src/main.rs`
    *   `crates/llm-map/Cargo.toml`
    *   `crates/llm-map/README.md`
    *   `crates/llm-map/tests/cli.rs` (Initial integration test file)
3.  **Update Workspace Manifest:**
    *   Add `crates/llm-map` to the root `Cargo.toml`'s `workspace.members`.

**Phase 2: Dependency Configuration**

1.  **Edit `crates/llm-map/Cargo.toml`:**
    *   Define `[package]` section.
    *   Add main dependencies `[dependencies]`:
        *   `rust-genai = "0.6"`
        *   `tokio = { version = "1", features = ["full"] }`
        *   `clap = { version = "4", features = ["derive"] }`
        *   `futures = "0.3"`
        *   `anyhow = "1.0"`
        *   `dotenvy = "0.15"`
        *   `tracing = "0.1"` (Optional)
        *   `tracing-subscriber = { version = "0.3", features = ["env-filter"] }` (Optional)
    *   Add development dependencies `[dev-dependencies]`:
        *   `assert_cmd = "2.0"` (CLI testing)
        *   `predicates = "3.0"` (Assertions on output/status)
        *   `tempfile = "3"` (Temporary files/dirs for tests)
        *   `wiremock = "0.6"` (Mocking the Gemini API)
        *   `serde_json = "1.0"` (For constructing mock responses)

**Phase 3: Initial Implementation & Test Setup (`src/main.rs`, `tests/cli.rs`)**

1.  **Basic CLI Structure (`src/main.rs`):**
    *   Define the `clap::Parser` struct for arguments (`prompt`, `model`, `files`, `concurrency`).
    *   Create a basic `#[tokio::main] async fn main() -> anyhow::Result<()>` function that parses args and prints them (or just exits successfully).
2.  **Basic CLI Tests (`tests/cli.rs`):**
    *   Use `assert_cmd::Command` to run the compiled binary.
    *   Test basic invocation (`cmd.assert().success();`).
    *   Test argument parsing: provide various arguments (`-p`, `-m`, file paths) and assert success. Test missing required arguments and assert failure.
3.  **Mock API Test Setup (`tests/cli.rs`):**
    *   Add helper functions or a test module setup using `wiremock::MockServer::start().await`.
    *   Define mock responses for Gemini API calls (e.g., a successful text generation response, an error response). Use `wiremock::{Mock, ResponseTemplate, Matchers}`.
    *   Mount mocks onto the server (`Mock::given(...).respond_with(...).mount(&mock_server).await;`).
    *   **Crucially:** Design how the application (`src/main.rs`) will know to use the mock server's URL instead of the real Gemini API during tests. An environment variable (e.g., `GEMINI_API_ENDPOINT_OVERRIDE`) checked within `main.rs` is a common pattern.
4.  **API Client Initialization (`src/main.rs`):**
    *   Implement logic to read `GEMINI_API_KEY` (using `dotenvy` and `std::env::var`).
    *   Implement logic to check for `GEMINI_API_ENDPOINT_OVERRIDE` and use that URL for the `genai::Client` if present; otherwise, use the default Gemini endpoint. Handle potential errors (missing key when override isn't set).

**Phase 4: Core Logic Implementation & Testing (Iterative)**

1.  **File Processing Logic (`src/main.rs` or refactored module):**
    *   Implement the function to read a file asynchronously (`async fn read_file_content(...)`).
    *   Write unit/integration tests for file reading (using `tempfile`).
2.  **API Interaction Logic (`src/main.rs` or refactored module):**
    *   Implement the function to call the Gemini API (`async fn call_gemini(...)`) using the configured `genai::Client`.
    *   Write integration tests using the `wiremock` server (`tests/cli.rs`):
        *   Test successful API call and response handling.
        *   Test API error handling.
3.  **Output Formatting (`src/main.rs` or refactored module):**
    *   Implement the logic to format the output string as specified (with `--- START FILE: ...`, `|` prefixes, `--- END FILE: ...`).
    *   Write unit tests for the formatting logic.
4.  **Concurrency & Output Synchronization (`src/main.rs`):**
    *   Implement the main loop using `futures::stream::iter`, `map`, `buffer_unordered`.
    *   Implement the `tokio::sync::Mutex` around `stdout` for writing results.
    *   Write integration tests (`tests/cli.rs`) using the mock server (potentially with programmed delays in responses via `ResponseTemplate::new(...).set_delay(...)`) to verify:
        *   Correct output for multiple successful files.
        *   Correct error reporting for failed files mixed with successful ones.
        *   Absence of interleaved output from concurrent operations.

**Phase 5: Refinement & Documentation**

1.  **Error Handling:** Ensure consistent and informative error messages are printed to `stderr`.
2.  **README:** Update `crates/llm-map/README.md` with detailed usage, setup (`GEMINI_API_KEY`), examples, and mention of the testing setup (e.g., `GEMINI_API_ENDPOINT_OVERRIDE`).

**Mermaid Diagram (Simplified Flow):**

```mermaid
graph TD
    A[Start] --> B(Parse Args);

    subgraph Tests_Args [Test Argument Parsing]
        TA1(Test Valid Args) --> B;
        TA2(Test Invalid Args) --> B;
    end

    B --> C{API Key & Endpoint Override?};

    subgraph Tests_Config [Test Configuration]
        TC1(Test Missing API Key) --> C;
        TC2(Test API Endpoint Override) --> C;
    end

    C -- Yes --> D(Init GenAI Client);
    C -- No --> X(Error: Config Issue);

    D --> E(Create File Stream);
    E --> F[Process Files Concurrently];

    subgraph F [Concurrent Processing (Buffered)]
        G(Read File) --> H(Call Gemini API / Mock API) --> I(Format Output);
    end

    subgraph Tests_Core [Test Core Logic]
        TG(Test File Reading) --> G;
        TH(Test API Call - Mocked) --> H;
        TI(Test Output Formatting) --> I;
    end

    F --> J{Collect Results};
    J --> K[Serialize Output];

    subgraph K [Serialized Output (Mutex)]
        L(Lock Stdout) --> M(Print Result) --> N(Unlock Stdout);
    end

    subgraph Tests_Integration [Test Integration & Concurrency]
        TI1(Test Full Success Flow - Mocked) --> K;
        TI2(Test Mixed Success/Error - Mocked) --> K;
        TI3(Test Concurrency & Output Order - Mocked) --> K;
    end

    K --> O(End);

    B -- Error --> Y(Error: Invalid Args);
    F -- File/API Error --> P(Log Error to Stderr);
    P --> J;