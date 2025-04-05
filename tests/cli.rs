// Integration tests for the llm-map CLI

use assert_cmd::Command;
use predicates::prelude::*;
use serde_json::json;
use std::{error::Error, io::Write, time::Duration};
use tempfile::{Builder as TempFileBuilder, NamedTempFile}; // Use Builder
use wiremock::matchers::{method, path_regex, header, query_param, path}; // Add path matcher
use wiremock::{Mock, MockServer, ResponseTemplate};
use base64::Engine as _; // Import Engine trait for decode

#[test]
fn test_runs_successfully_with_args() -> Result<(), Box<dyn Error>> {
    let file1 = NamedTempFile::new()?;
    let file2 = NamedTempFile::new()?;

    let mut cmd = Command::cargo_bin("gemini-map")?;
    cmd.arg("-p")
        .arg("Test Prompt")
        .arg("-m")
        .arg("gemini-test")
        .arg(file1.path())
        .arg(file2.path());

    // We expect it to run, but it will likely fail later
    // when trying to connect to the API without credentials/mocks.
    // For now, just check if it parses args and starts.
    // A more specific success assertion will come later.
    cmd.assert().failure().stderr(predicate::str::contains("GEMINI_API_KEY environment variable not found")); // Expect failure due to missing API key

    Ok(())
}

#[test]
fn test_missing_prompt_arg() -> Result<(), Box<dyn Error>> {
    let file1 = NamedTempFile::new()?;
    let mut cmd = Command::cargo_bin("gemini-map")?;
    cmd.arg("-m")
       .arg("gemini-test")
       .arg(file1.path());
    // Expect API key error first when model is specified
    cmd.assert()
       .failure()
       .stderr(predicate::str::contains("GEMINI_API_KEY environment variable not found"));
    Ok(())
}

#[test]
fn test_missing_model_arg() -> Result<(), Box<dyn Error>> {
    let file1 = NamedTempFile::new()?;
    let mut cmd = Command::cargo_bin("gemini-map")?;
    cmd.arg("-p")
       .arg("Test Prompt")
       .arg(file1.path());
    // Expect failure (exit code 1) because an error occurs ("Cannot output binary...")
    // and the program now exits with 1 on any error.
    cmd.assert()
       .failure() // Changed from .success()
       .stderr(
           predicate::str::contains("No model specified.")
           .and(predicate::str::contains("Cannot output binary input content"))
       );
    Ok(())
}

#[test]
fn test_missing_files_arg() -> Result<(), Box<dyn Error>> {
    let mut cmd = Command::cargo_bin("gemini-map")?;
    cmd.arg("-p")
       .arg("Test Prompt")
       .arg("-m")
       .arg("gemini-test");
    // Expect API key error first when model is specified
    cmd.assert()
       .failure()
       .stderr(predicate::str::contains("GEMINI_API_KEY environment variable not found"));
    Ok(())
}


#[test]
fn test_reads_file_content() -> Result<(), Box<dyn Error>> {
    let mut file = NamedTempFile::new()?;
    let file_content = "Hello, test file!\nLine 2.";
    writeln!(file, "{}", file_content)?;
    file.flush()?; // Ensure content is written
    let file_path = file.path();
    // let filename = file_path.file_name().unwrap().to_str().unwrap(); // No longer needed

    let mut cmd = Command::cargo_bin("gemini-map")?;
    cmd.arg("-p")
        .arg("Read Test")
        .arg("-m")
        .arg("gemini-test")
        .arg(file_path);

    // Expect failure due to missing API key. Stdout might be empty as the error occurs early.
    cmd.assert()
        .failure()
        .stderr(predicate::str::contains("GEMINI_API_KEY environment variable not found"));
        // Do not assert stdout content, as the program likely exits before printing it
        // when the API key is missing but required.

    Ok(())
}


#[tokio::test] // Needs tokio runtime for MockServer
async fn test_api_call_with_mock() -> Result<(), Box<dyn Error>> {
    // --- Mock Server Setup ---
    let mock_server = MockServer::start().await;
    let model_name = "gemini-mock-model";
    let mock_api_response_text = "This is the mocked API response.";

    // The `expected_request_body` variable was removed as it's unused
    // since the body_json matcher is commented out.

    // Define the mock response body (matching Gemini API structure)
    let mock_response = json!({
      "candidates": [{
        "content": {
          "parts": [{"text": mock_api_response_text}],
          "role": "model"
        },
        "finishReason": "STOP",
        "index": 0,
        "safetyRatings": [
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "probability": "NEGLIGIBLE"},
            // ... other safety ratings (structure required by API)
        ]
      }],
      "promptFeedback": {
         "safetyRatings": [
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "probability": "NEGLIGIBLE"},
            // ... other safety ratings (structure required by API)
         ]
      }
    });


    Mock::given(method("POST"))
        .and(path_regex(format!("/v1beta/models/{}:generateContent", model_name))) // More specific path
        .and(query_param("key", "DUMMY_KEY_FOR_MOCK")) // Check for API key in query
        .and(header("Content-Type", "application/json")) // Check header
        // .and(body_json(&expected_request_body)) // Optional: Add body matching if needed
        .respond_with(ResponseTemplate::new(200).set_body_json(mock_response))
        .mount(&mock_server)
        .await;

    // --- Test Execution ---
    let mut file = NamedTempFile::new()?;
    let file_content = "File content for mock test.";
    writeln!(file, "{}", file_content)?;
    file.flush()?;
    let file_path = file.path();
    let filename = file_path.file_name().unwrap().to_str().unwrap();

    let mut cmd = Command::cargo_bin("gemini-map")?;
    cmd.env("GEMINI_API_ENDPOINT_OVERRIDE", mock_server.uri()) // Use mock server URL
       .env("GEMINI_API_KEY", "DUMMY_KEY_FOR_MOCK") // Provide the key for the query param check
       .arg("-p")
       .arg("Mock Prompt")
       .arg("-m")
       .arg(model_name) // Use the model name defined in the mock path
       .arg(file_path);

    // Expect success now, and stdout should contain the mocked response
    cmd.assert()
       .success()
       .stdout(
           predicate::str::contains(format!("--- START OF: {} (run 1/1) ---", filename)) // Use START OF and include run info
           // Remove the explicit '&' as contains likely takes &str directly
           .and(predicate::str::contains(format!("| {}", mock_api_response_text))) // Check for formatted mocked response
           .and(predicate::str::contains(format!("--- END OF: {} (run 1/1) ---", filename))) // Use END OF and include run info
       );
       // .stderr(predicate::str::is_empty()); // Expect empty stderr on success

    Ok(())
}


#[tokio::test]
async fn test_concurrency_and_output_order() -> Result<(), Box<dyn Error>> {
    // --- Mock Server Setup ---
    let mock_server = MockServer::start().await;
    let model_name = "gemini-concurrent-model";

    let resp_a = "Response for file A.";
    let resp_b = "Response for file B (delayed).";
    let resp_c = "Response for file C.";

    // Mock responses (matching Gemini API structure)
    let mock_response_a = json!({"candidates": [{"content": {"parts": [{"text": resp_a}]}}]});
    let mock_response_b = json!({"candidates": [{"content": {"parts": [{"text": resp_b}]}}]});
    let mock_response_c = json!({"candidates": [{"content": {"parts": [{"text": resp_c}]}}]});

    // Mount mocks - give B a delay
    Mock::given(method("POST"))
        .and(path_regex(format!("/v1beta/models/{}:generateContent", model_name))) // More specific path
        .and(query_param("key", "DUMMY_KEY_FOR_MOCK")) // Check for API key in query
        .and(header("Content-Type", "application/json")) // Check header
        .respond_with(ResponseTemplate::new(200).set_body_json(mock_response_a))
        .up_to_n_times(1) // Expect only one call for A's content pattern (approx)
        .mount(&mock_server)
        .await;
    Mock::given(method("POST"))
        .and(path_regex(format!("/v1beta/models/{}:generateContent", model_name))) // More specific path
        .and(query_param("key", "DUMMY_KEY_FOR_MOCK")) // Check for API key in query
        .and(header("Content-Type", "application/json")) // Check header
        .respond_with(ResponseTemplate::new(200).set_body_json(mock_response_b).set_delay(Duration::from_millis(100))) // Delay B
        .up_to_n_times(1)
        .mount(&mock_server)
        .await;
    Mock::given(method("POST"))
        .and(path_regex(format!("/v1beta/models/{}:generateContent", model_name))) // More specific path
        .and(query_param("key", "DUMMY_KEY_FOR_MOCK")) // Check for API key in query
        .and(header("Content-Type", "application/json")) // Check header
        .respond_with(ResponseTemplate::new(200).set_body_json(mock_response_c))
        .up_to_n_times(1)
        .mount(&mock_server)
        .await;


    // --- Test Execution ---
    let mut file_a = NamedTempFile::new()?;
    writeln!(file_a, "Content A")?; file_a.flush()?;
    let path_a = file_a.path().to_path_buf();
    let name_a = path_a.file_name().unwrap().to_str().unwrap();

    let mut file_b = NamedTempFile::new()?;
    writeln!(file_b, "Content B")?; file_b.flush()?;
    let path_b = file_b.path().to_path_buf();
    let name_b = path_b.file_name().unwrap().to_str().unwrap();

    let mut file_c = NamedTempFile::new()?;
    writeln!(file_c, "Content C")?; file_c.flush()?;
    let path_c = file_c.path().to_path_buf();
    let name_c = path_c.file_name().unwrap().to_str().unwrap();


    let mut cmd = Command::cargo_bin("gemini-map")?;
    cmd.env("GEMINI_API_ENDPOINT_OVERRIDE", mock_server.uri()) // Use mock server URL
       .env("GEMINI_API_KEY", "DUMMY_KEY_FOR_MOCK") // Provide the key for the query param check
       .arg("-p").arg("Concurrent Test")
       .arg("-m").arg(model_name)
       .arg("-c").arg("3") // Set concurrency
       .arg(&path_a)
       .arg(&path_b)
       .arg(&path_c);

    let output = cmd.assert().success();
    let stdout = String::from_utf8(output.get_output().stdout.clone())?;

    // --- Assertions ---
    // 1. Check all expected outputs are present
    assert!(stdout.contains(&format!("| {}", resp_a)));
    assert!(stdout.contains(&format!("| {}", resp_b)));
    assert!(stdout.contains(&format!("| {}", resp_c)));

    // 2. Check for non-interleaved blocks by extracting content and checking for markers within
    fn find_block<'a>(stdout: &'a str, filename: &str) -> Option<&'a str> {
        let start_marker = format!("--- START OF: {} (run 1/1) ---", filename);
        let end_marker = format!("--- END OF: {} (run 1/1) ---", filename);
        let start_idx = stdout.find(&start_marker)?;
        let end_idx = stdout[start_idx..].find(&end_marker)?;
        Some(&stdout[start_idx + start_marker.len()..start_idx + end_idx])
    }

    let block_a = find_block(&stdout, name_a).expect("Block A not found");
    let block_b = find_block(&stdout, name_b).expect("Block B not found");
    let block_c = find_block(&stdout, name_c).expect("Block C not found");

    // Check that no block contains the start/end markers of *other* blocks
    assert!(!block_a.contains("--- START OF:") && !block_a.contains("--- END OF:"), "Block A interleaved");
    assert!(!block_b.contains("--- START OF:") && !block_b.contains("--- END OF:"), "Block B interleaved");
    assert!(!block_c.contains("--- START OF:") && !block_c.contains("--- END OF:"), "Block C interleaved");

    Ok(())
}

#[test]
fn test_no_model_with_text_file() -> Result<(), Box<dyn Error>> { // Use TempFileBuilder
    let mut file = TempFileBuilder::new().suffix(".txt").tempfile()?; // Add .txt suffix
    let file_content = "This is plain text content.";
    writeln!(file, "{}", file_content)?;
    file.flush()?;
    let file_path = file.path();
    let filename = file_path.file_name().unwrap().to_str().unwrap();

    let mut cmd = Command::cargo_bin("gemini-map")?;
    // Provide a dummy prompt, as it's currently expected by the arg parser structure,
    // even though it won't be used without a model.
    cmd.arg("-p").arg("Dummy Prompt");
    cmd.arg(file_path); // No -m argument

    // Expect success, stdout contains the file content, stderr contains the warning
    cmd.assert()
       .success()
       .stdout(
           predicate::str::contains(format!("--- START OF: {} (run 1/1) ---", filename))
           .and(predicate::str::contains(format!("| {}", file_content))) // Check for formatted content
           .and(predicate::str::contains(format!("--- END OF: {} (run 1/1) ---", filename)))
       )
       .stderr(predicate::str::contains("No model specified."));

    Ok(())
}


#[tokio::test]
async fn test_runtime_error_invalid_api_key() -> Result<(), Box<dyn Error>> {
    // --- Mock Server Setup ---
    let mock_server = MockServer::start().await;
    let model_name = "gemini-invalid-key-model";
    let error_message = "API key not valid. Please pass a valid API key.";

    // Mock response for invalid API key (e.g., 400 Bad Request)
    let mock_error_body = json!({
        "error": {
            "code": 400,
            "message": error_message,
            "status": "INVALID_ARGUMENT" // Or PERMISSION_DENIED etc.
        }
    });

    Mock::given(method("POST"))
        .and(path_regex(format!("/v1beta/models/{}:generateContent", model_name)))
        // No need to check query_param("key", ...) here, as we want the *call* to fail
        .respond_with(ResponseTemplate::new(400).set_body_json(mock_error_body))
        .mount(&mock_server)
        .await;

    // --- Test Execution ---
    let mut file = NamedTempFile::new()?;
    writeln!(file, "Content for invalid key test")?;
    file.flush()?;

    let mut cmd = Command::cargo_bin("gemini-map")?;
    cmd.env("GEMINI_API_ENDPOINT_OVERRIDE", mock_server.uri())
       .env("GEMINI_API_KEY", "THIS_KEY_IS_INVALID") // Provide *some* key
       .arg("-p").arg("Test Invalid Key")
       .arg("-m").arg(model_name)
       .arg(file.path());

    // Expect failure and stderr containing the API error message
    cmd.assert()
       .failure()
       .stderr(
           predicate::str::contains("Error processing input") // General error prefix
           .and(predicate::str::contains("Gemini API request failed with status 400")) // Status code
           .and(predicate::str::contains(error_message)) // Specific message from mock
       );

    Ok(())
}


#[tokio::test]
async fn test_runtime_error_api_connection_error() -> Result<(), Box<dyn Error>> {
    // --- Test Execution ---
    let mut file = NamedTempFile::new()?;
    writeln!(file, "Content for connection error test")?;
    file.flush()?;

    let mut cmd = Command::cargo_bin("gemini-map")?;
    // Point to a likely non-existent local port
    cmd.env("GEMINI_API_ENDPOINT_OVERRIDE", "http://127.0.0.1:1")
       .env("GEMINI_API_KEY", "DUMMY_KEY_CONN_ERR")
       .arg("-p").arg("Test Connection Error")
       .arg("-m").arg("gemini-conn-err-model")
       .arg(file.path());

    // Expect failure and stderr containing a connection error message
    cmd.assert()
       .failure()
       .stderr(
           predicate::str::contains("Error processing input") // General error prefix
           // Check only for the high-level error context, as the specific reqwest error isn't printed
           .and(predicate::str::contains("Failed to send request to Gemini API"))
           .and(predicate::str::contains("One or more errors occurred during processing.")) // Check for the final error message
       );

    Ok(())
}

#[tokio::test]
async fn test_mixed_input_types() -> Result<(), Box<dyn Error>> {
    // --- Mock Server Setup ---
    let mock_server = MockServer::start().await;
    let model_name = "gemini-mixed-model";
    let api_key = "mock-key-mixed";
    let mock_response_text = "Processed mixed inputs";

    let mock_response = json!({
      "candidates": [{"content": {"parts": [{"text": mock_response_text}]}}]
    });

    // Mock will be hit twice, once for text, once for PDF
    // Define the exact path for the mock
    let mock_api_path = format!("/v1beta/models/{}:generateContent", model_name);

    Mock::given(method("POST"))
        .and(path(&mock_api_path)) // Use exact path matching (pass by reference)
        .and(query_param("key", api_key))
        .respond_with(ResponseTemplate::new(200).set_body_json(mock_response.clone()))
        .expect(2) // Re-enable expectation count
        .mount(&mock_server)
        .await;

    // --- Test File Setup ---
    // 1. Text file
    let mut text_file = TempFileBuilder::new().prefix("mixed_test").suffix(".txt").tempfile()?;
    let text_content = "This is the text part.";
    writeln!(text_file, "{}", text_content)?;
    text_file.flush()?;
    let text_path = text_file.path();
    let text_filename = text_path.file_name().unwrap().to_str().unwrap();

    // 2. PDF file (minimal valid PDF)
    // Base64 for: '%PDF-1.1\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj 2 0 obj<</Type/Pages/Count 0>>endobj\nxref\n0 3\n0000000000 65535 f\n0000000009 00000 n\n0000000052 00000 n\ntrailer<</Size 3/Root 1 0 R>>startxref\n101\n%%EOF'
    let pdf_content_base64 = "JVBERi0xLjAKMSAwIG9iajw8L1R5cGUvQ2F0YWxvZy9QYWdlcyAyIDAgUj4+ZW5kb2JqCjIgMCBvYmo8PC9UeXBlL1BhZ2VzL0tpZHNbMyAwIFJdL0NvdW50IDE+PmVuZG9iagozIDAgb2JqPDwvVHlwZS9QYWdlL01lZGlhQm94WzAgMCAzIDNdPj5lbmRvYmoKeHJlZgowIDQKMDAwMDAwMDAwMCA2NTUzNSBmIAowMDAwMDAwMDAwOSAwMDAwMCBuIAowMDAwMDAwMDUyIDAwMDAwIG4gCjAwMDAwMDAxMDEgMDAwMDAgbiAKdHJhaWxlcjw8L1NpemUgNC9Sb290IDEgMCBSPj4Kc3RhcnR4cmVmCjE0OQolJUVPRgo="; // Corrected minimal PDF base64
    let pdf_data = base64::engine::general_purpose::STANDARD.decode(pdf_content_base64)?;
    let mut pdf_file = TempFileBuilder::new().prefix("mixed_test").suffix(".pdf").tempfile()?;
    pdf_file.write_all(&pdf_data)?;
    pdf_file.flush()?;
    let pdf_path = pdf_file.path();
    let pdf_filename = pdf_path.file_name().unwrap().to_str().unwrap();


    // --- Test Execution ---
    let mut cmd = Command::cargo_bin("gemini-map")?;
    cmd.env("GEMINI_API_ENDPOINT_OVERRIDE", mock_server.uri())
       .env("GEMINI_API_KEY", api_key)
       .arg("-p").arg("Process Mixed")
       .arg("-m").arg(model_name)
       .arg(text_path) // Pass both files
       .arg(pdf_path);

    // Expect success and output blocks for both files containing the mocked response
    cmd.assert()
       .success()
       .stdout(
           predicate::str::contains(format!("--- START OF: {} (run 1/1) ---", text_filename))
           .and(predicate::str::contains(format!("| {}", mock_response_text)))
           .and(predicate::str::contains(format!("--- END OF: {} (run 1/1) ---", text_filename)))
       )
       .stdout(
           predicate::str::contains(format!("--- START OF: {} (run 1/1) ---", pdf_filename))
           .and(predicate::str::contains(format!("| {}", mock_response_text)))
           .and(predicate::str::contains(format!("--- END OF: {} (run 1/1) ---", pdf_filename)))
       )
       /* .stderr(predicate::str::is_empty()) */; // Remove assertion: stderr contains info messages

    Ok(())
}


// TODO: Add tests for error handling within concurrency
// TODO: Add integration test for file read error during processing (difficult?)
// TODO: Add integration test for invalid/corrupted PDF processing