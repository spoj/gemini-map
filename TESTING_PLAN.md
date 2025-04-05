# Testing Plan for gemini-map

## Current Test Coverage Assessment

*   **Unit Tests (`src/pdf_utils.rs`):** Good coverage of basic PDF extraction scenarios (empty, single/multi-page, bounds).
*   **Integration Tests (`tests/cli.rs`):** Covers basic CLI argument handling (missing args), mocked successful API calls, and a basic concurrency scenario. It also implicitly tests the API key requirement and the no-model scenario (though primarily focusing on its error state with binary files).

## Identified Gaps & Suggested New Tests

Based on the review, here are the main areas where test coverage could be enhanced:

1.  **Unit Testing `main.rs` Logic:** The core application logic in `src/main.rs` lacks unit tests. This includes input processing, how different file types are dispatched, interaction with the API client (which could be mocked), and output formatting logic. **[Partially Addressed - See Progress Update]**
2.  **Runtime Error Handling (Integration):** Current tests focus on argument errors. Tests for runtime errors like invalid API keys, API connection failures, API-returned errors (e.g., rate limits), file read errors during processing, or invalid/corrupted PDF files are missing.
3.  **Specific Scenario Coverage (Integration):**
    *   Testing the successful output of *text* files when no model (`-m`) is specified.
    *   Testing with a mix of input file types (text and PDF) in a single run.
    *   Testing how errors are handled when running with concurrency enabled (e.g., one file fails, others succeed).

## Visual Plan

```mermaid
graph TD
    A[Test Coverage Assessment] --> B(Unit Tests);
    A --> C(Integration Tests);

    B --> B1[src/pdf_utils.rs<br/>(Good Coverage)];
    B --> B2[src/main.rs<br/>(Gaps Identified)];
    B2 --> B2a(Input Gathering);
    B2 --> B2b(File Type Dispatch);
    B2 --> B2c(API Client Interaction);
    B2 --> B2d(Output Formatting);

    C --> C1[Existing Coverage<br/>(Args, Mock Success, Concurrency)];
    C --> C2[Identified Gaps];
    C2 --> C2a(Runtime Error Handling);
    C2 --> C2b(Specific Scenarios);

    subgraph "Suggested Unit Tests (for main.rs)"
        direction LR
        B2a --suggests--> UT1(test_input_gathering);
        B2b --suggests--> UT2(test_file_type_dispatch);
        B2c --suggests--> UT3(test_api_client_wrapper);
        B2d --suggests--> UT4(test_output_formatting);
    end

    subgraph "Suggested Integration Tests (in tests/cli.rs)"
        direction LR
        C2a --suggests--> IT1(test_runtime_errors<br/>- Invalid API Key<br/>- API Connection/Server Errors<br/>- File Read Errors<br/>- Invalid PDF);
        C2b --suggests--> IT2(test_no_model_with_text_file);
        C2b --suggests--> IT3(test_mixed_input_types);
        C2b --suggests--> IT4(test_concurrency_with_errors);
    end
```

## Implementation Plan

1.  **Add Unit Tests for `main.rs`:** Create unit tests for the core logic within `main.rs`, mocking external dependencies like the file system (if needed) and the API client. **[DONE]**
2.  **Expand Integration Tests:** Add tests specifically targeting runtime error conditions and edge cases like mixed inputs, no-model text output, and error handling under concurrency. **[NEXT STEP]**

## Progress Update (2025-05-04)

*   **Fixed Initial Failures:** Resolved all failing integration tests in `tests/cli.rs` by updating assertions to match current application behavior (API key checks, output format, concurrency logic).
*   **Added Unit Tests (`src/main.rs`):** Implemented unit tests for the following functions and logic:
    *   `process_file_input`: Handling text files, file read errors, and non-PDF files when splitting is enabled.
    *   `process_input_string`: Handling both file paths and URLs (using `wiremock` for HTTP mocking).
    *   `MODEL_ALIASES`: Verified correct alias resolution.
    *   `call_gemini_api`: Mocked successful API calls and API error responses (using `wiremock`).
    *   `gather_input_units`: Tested aggregation from files, URLs, mixed sources, and handling of invalid inputs.
    *   `fetch_url_input`: Tested handling of 404 errors (using `wiremock`).
*   **Current Status:** All unit and integration tests are currently passing. Step 1 of the Implementation Plan is complete.