// Integration tests for the llm-map CLI

use assert_cmd::Command; // Run programs
use predicates::prelude::*; // Used for writing assertions
use serde_json::json;
use std::{error::Error, io::Write, time::Duration}; // Added Duration
use tempfile::NamedTempFile;
use wiremock::matchers::{method, path_regex, header, query_param}; // Added header, query_param
use wiremock::{Mock, MockServer, ResponseTemplate};

#[test]
fn test_runs_successfully_with_args() -> Result<(), Box<dyn Error>> {
    // Create dummy files
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
    cmd.assert().failure().stderr(predicate::str::contains("GEMINI_API_KEY not found")); // Expect failure due to missing API key for now

    Ok(())
}

#[test]
fn test_missing_prompt_arg() -> Result<(), Box<dyn Error>> {
    let file1 = NamedTempFile::new()?;
    let mut cmd = Command::cargo_bin("gemini-map")?;
    cmd.arg("-m")
       .arg("gemini-test")
       .arg(file1.path());
    cmd.assert()
       .failure()
       .stderr(predicate::str::contains("required arguments were not provided"));
    Ok(())
}

#[test]
fn test_missing_model_arg() -> Result<(), Box<dyn Error>> {
    let file1 = NamedTempFile::new()?;
    let mut cmd = Command::cargo_bin("gemini-map")?;
    cmd.arg("-p")
       .arg("Test Prompt")
       .arg(file1.path());
    cmd.assert()
       .failure()
       .stderr(predicate::str::contains("required arguments were not provided"));
    Ok(())
}

#[test]
fn test_missing_files_arg() -> Result<(), Box<dyn Error>> {
    let mut cmd = Command::cargo_bin("gemini-map")?;
    cmd.arg("-p")
       .arg("Test Prompt")
       .arg("-m")
       .arg("gemini-test");
    cmd.assert()
       .failure()
       .stderr(predicate::str::contains("required arguments were not provided"));
    Ok(())
}


#[test]
fn test_reads_file_content() -> Result<(), Box<dyn Error>> {
    // Create a temp file with known content
    let mut file = NamedTempFile::new()?;
    let file_content = "Hello, test file!\nLine 2.";
    writeln!(file, "{}", file_content)?;
    file.flush()?; // Ensure content is written
    let file_path = file.path();
    let filename = file_path.file_name().unwrap().to_str().unwrap();

    let mut cmd = Command::cargo_bin("gemini-map")?;
    cmd.arg("-p")
        .arg("Read Test")
        .arg("-m")
        .arg("gemini-test")
        .arg(file_path);

    // Expect failure due to missing API key, but stdout should contain file content
    cmd.assert()
        .failure()
        .stdout(
            predicate::str::contains(format!("--- START FILE: {} ---", filename))
            .and(predicate::str::contains(file_content))
            .and(predicate::str::contains(format!("--- END FILE: {} ---", filename)))
        )
        .stderr(predicate::str::contains("GEMINI_API_KEY not found"));

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
            // ... other safety ratings
        ]
      }],
      "promptFeedback": {
         "safetyRatings": [
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "probability": "NEGLIGIBLE"},
            // ... other safety ratings
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
           predicate::str::contains(format!("--- START FILE: {} ---", filename))
           .and(predicate::str::contains(&format!("| {}", mock_api_response_text))) // Check for *formatted* mocked response
           .and(predicate::str::contains(format!("--- END FILE: {} ---", filename)))
       ); // End of stdout predicate chain
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

    // 2. Check for non-interleaved blocks using regex
    // This regex looks for the START marker, then any characters (*? non-greedy)
    // until the corresponding END marker, ensuring no other START marker appears in between.
    // It checks this for all three files. The order might vary due to buffer_unordered.
    let pattern_a = format!(r"(?s)--- START FILE: {} ---(.(?!START FILE))*?--- END FILE: {} ---", name_a, name_a);
    let pattern_b = format!(r"(?s)--- START FILE: {} ---(.(?!START FILE))*?--- END FILE: {} ---", name_b, name_b);
    let pattern_c = format!(r"(?s)--- START FILE: {} ---(.(?!START FILE))*?--- END FILE: {} ---", name_c, name_c);

    let re_a = regex::Regex::new(&pattern_a)?;
    let re_b = regex::Regex::new(&pattern_b)?;
    let re_c = regex::Regex::new(&pattern_c)?;

    assert!(re_a.is_match(&stdout), "Block for file A is interleaved or missing.");
    assert!(re_b.is_match(&stdout), "Block for file B is interleaved or missing.");
    assert!(re_c.is_match(&stdout), "Block for file C is interleaved or missing.");

    Ok(())
}


// TODO: Add tests for error handling within concurrency