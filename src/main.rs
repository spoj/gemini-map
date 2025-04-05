use anyhow::{Context, Result, anyhow, bail};
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64_STANDARD};
use clap::Parser;
use futures::stream::{self, StreamExt};
// Only Document is used directly in this file now
use lopdf::Document;
use reqwest::Client as ReqwestClient;
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    env,
    path::PathBuf,
    str,
    sync::{
        Arc, LazyLock,
        atomic::{AtomicBool, Ordering},
    },
};
use tokio::{
    fs,
    io::{AsyncWriteExt, stdout},
    sync::Mutex,
};
use url::Url;

mod pdf_utils;
#[derive(Serialize)]
struct Part {
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<String>,
    #[serde(rename = "inlineData", skip_serializing_if = "Option::is_none")]
    inline_data: Option<InlineData>,
}

#[derive(Serialize)]
struct InlineData {
    #[serde(rename = "mimeType")]
    mime_type: String,
    data: String,
}

#[derive(Serialize)]
struct Content {
    parts: Vec<Part>,
    role: String,
}

// Add this struct definition
#[derive(Serialize, Debug, Clone)]
struct GenerationConfig {
    temperature: f32,
}

#[derive(Serialize)]
struct GenerateContentRequest {
    contents: Vec<Content>,
    #[serde(rename = "generationConfig", skip_serializing_if = "Option::is_none")]
    generation_config: Option<GenerationConfig>, // Add this field
}

#[derive(Deserialize, Debug)]
struct GenerateContentResponse {
    candidates: Option<Vec<Candidate>>,
}

#[derive(Deserialize, Debug)]
struct Candidate {
    content: Option<ContentResponse>,
}

#[derive(Deserialize, Debug)]
struct ContentResponse {
    parts: Option<Vec<PartResponse>>,
}

#[derive(Deserialize, Debug)]
struct PartResponse {
    text: Option<String>,
}

#[derive(Debug, Clone)]
struct InputUnit {
    identifier: String,
    data: Vec<u8>,
    mime_type: mime::Mime,
}
#[derive(Debug, Clone)]
enum WorkItem {
    ProcessInput(InputUnit),
    ProcessPrompt,
}

static MODEL_ALIASES: LazyLock<HashMap<&'static str, &'static str>> = LazyLock::new(|| {
    let mut m = HashMap::new();
    m.insert("flash-think", "gemini-2.0-flash-thinking-exp-01-21");
    m.insert("pro", "gemini-2.5-pro-exp-03-25");
    m.insert("flash", "gemini-2.0-flash");
    m
});

/// Apply a Gemini prompt to multiple files concurrently.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// The prompt to send to the Gemini model along with the file content. Optional.
    #[arg(short, long)]
    prompt: Option<String>,

    /// The specific Gemini model to use (e.g., "gemini-pro", "pro", "flash"). Optional.
    /// If omitted, outputs file content directly without calling the LLM.
    #[arg(short, long)]
    model: Option<String>,

    /// The maximum number of inputs to process concurrently.
    #[arg(short, long, default_value_t = 5)]
    concurrency: usize,

    /// Number of times to repeat the prompt for each input unit.
    #[arg(short, long, default_value_t = 1)]
    repeats: usize,

    /// List of input files or URLs. Optional if --prompt is provided.
    inputs: Vec<String>,

    /// Split PDF files into individual pages (rendered as PNGs) instead of processing the whole file. Requires PDFium library.
    #[arg(short, long, default_value_t = false)]
    split_pdf: bool,

    /// Controls randomness: lower values are more deterministic, higher values more creative. Optional.
    #[arg(short, long)]
    temperature: Option<f32>,
}

/// Fetches content from a URL and creates an InputUnit.
async fn fetch_url_input(url: Url, client: &ReqwestClient) -> Result<Option<InputUnit>> {
    let identifier = url.to_string();
    eprintln!("Fetching URL: {}", identifier);
    match client.get(url).send().await {
        Ok(response) => {
            if response.status().is_success() {
                let mime_type = response
                    .headers()
                    .get(reqwest::header::CONTENT_TYPE)
                    .and_then(|val| val.to_str().ok())
                    .and_then(|s| s.parse::<mime::Mime>().ok())
                    .or_else(|| mime_guess::from_path(identifier.as_str()).first())
                    .unwrap_or(mime::APPLICATION_OCTET_STREAM);

                match response.bytes().await {
                    Ok(data) => Ok(Some(InputUnit {
                        identifier,
                        data: data.to_vec(),
                        mime_type,
                    })),
                    Err(e) => {
                        eprintln!("Error reading bytes from URL {}: {}", identifier, e);
                        Ok(None)
                    }
                }
            } else {
                eprintln!(
                    "Error fetching URL {}: Status {}",
                    identifier,
                    response.status()
                );
                Ok(None)
            }
        }
        Err(e) => {
            eprintln!("Error sending request to URL {}: {}", identifier, e);
            Ok(None)
        }
    }
}
// Removed dead function: get_inherited_attributes

/// Splits a PDF document into single-page PDF documents using the `pdf_utils` module.
fn split_pdf_lopdf(data: &[u8], identifier: &str) -> Result<Vec<InputUnit>> {
    let mut page_units = Vec::new();

    // We still need to load the doc once to get the page count.
    // Alternatively, pdf_utils could return page count, but this is simpler for now.
    let temp_doc = Document::load_mem(data).context("Failed to load PDF data to get page count")?;
    let page_count = temp_doc.get_pages().len();
    if page_count == 0 {
        eprintln!("Warning: PDF '{}' has no pages.", identifier);
        return Ok(page_units);
    }
    drop(temp_doc);

    eprintln!(
        "Splitting PDF with pdf_utils: {} ({} pages)",
        identifier, page_count
    );

    // Iterate from 1 to page_count (inclusive) as extract_page_from_pdf expects 1-based index
    for page_num_u32 in 1..=(page_count as u32) {
        let page_num = page_num_u32 as usize; // Keep usize for formatting consistency if needed
        let page_identifier = format!("{} (page {}/{})", identifier, page_num, page_count);
        eprintln!("Processing page: {}", page_identifier);

        match pdf_utils::extract_page_from_pdf(data, page_num_u32) {
            Ok(page_data) => {
                page_units.push(InputUnit {
                    identifier: page_identifier,
                    data: page_data,
                    mime_type: mime::APPLICATION_PDF,
                });
            }
            Err(e) => {
                eprintln!(
                    "Error extracting page {} of '{}': {:?}",
                    page_num, identifier, e
                );
                // Decide whether to continue or bail. Let's continue for now.
                // If you want to stop on the first error, return Err(e) here.
            }
        }
    }
    Ok(page_units)
}

/// Reads a file and creates InputUnits, potentially splitting PDFs.
async fn process_file_input(
    path: PathBuf,
    identifier: String,
    split_pdf: bool,
) -> Result<Vec<InputUnit>> {
    match fs::read(&path).await {
        Ok(data) => {
            let mime_type = mime_guess::from_path(&path).first_or(mime::APPLICATION_OCTET_STREAM);

            if mime_type == mime::APPLICATION_PDF && split_pdf {
                split_pdf_lopdf(&data, &identifier)
                    .map_err(|e| anyhow!("Failed to split PDF '{}' with lopdf: {}", identifier, e))
            } else {
                Ok(vec![InputUnit {
                    identifier,
                    data,
                    mime_type,
                }])
            }
        }
        Err(e) => {
            eprintln!("Error reading file {}: {}", identifier, e);
            Ok(Vec::new())
        }
    }
}

/// Processes a single input string (URL or file path).
async fn process_input_string(
    input_str: &str,
    client: &ReqwestClient,
    split_pdf: bool,
) -> Result<Vec<InputUnit>> {
    match Url::parse(input_str) {
        Ok(url) => fetch_url_input(url, client)
            .await
            .map(|opt| opt.map_or(Vec::new(), |unit| vec![unit])),
        Err(_) => {
            let path = PathBuf::from(input_str);
            let identifier = path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or(input_str)
                .to_string();
            process_file_input(path, identifier, split_pdf).await
        }
    }
}

// Processor struct to hold context for processing work items
#[derive(Clone)] // Clone is cheap due to Arcs
struct WorkItemProcessor {
    client: Option<Arc<ReqwestClient>>,
    api_key: Option<Arc<String>>,
    base_url: Option<Arc<String>>,
    model_name: Option<String>,
    prompt: Option<String>,
    temperature: Option<f32>,
}

impl WorkItemProcessor {
    /// Creates a new WorkItemProcessor with the necessary context.
    fn new(
        client: Option<Arc<ReqwestClient>>,
        api_key: Option<Arc<String>>,
        base_url: Option<Arc<String>>,
        model_name: Option<String>,
        prompt: Option<String>,
        temperature: Option<f32>,
    ) -> Self {
        Self {
            client,
            api_key,
            base_url,
            model_name,
            prompt,
            temperature,
        }
    }

    /// Processes a single work item using the stored context.
    async fn process(
        &self,               // Takes self by reference
        work_item: WorkItem, // Takes ownership of the WorkItem
        run_idx: usize,
        total_runs: usize,
    ) -> Result<(String, String), (String, anyhow::Error)> {
        // Returns Ok((id, content)) or Err((id, error))
        match work_item {
            WorkItem::ProcessInput(unit) => {
                let base_identifier = unit.identifier.clone();
                let identifier =
                    format!("{} (run {}/{})", base_identifier, run_idx + 1, total_runs);

                let res = match &self.model_name {
                    Some(model_name) => {
                        // Use context from self
                        let client = self
                            .client
                            .as_ref()
                            .expect("API client should be initialized");
                        let api_key = self
                            .api_key
                            .as_ref()
                            .expect("API key should be initialized");
                        let base_url = self
                            .base_url
                            .as_ref()
                            .expect("Base URL should be initialized");

                        let mut parts = Vec::new();
                        if let Some(prompt) = &self.prompt {
                            parts.push(Part {
                                text: Some(prompt.clone()),
                                inline_data: None,
                            });
                        }
                        let inline_data = InlineData {
                            mime_type: unit.mime_type.to_string(),
                            data: BASE64_STANDARD.encode(&unit.data),
                        };
                        parts.push(Part {
                            text: None,
                            inline_data: Some(inline_data),
                        });

                        let generation_config = self
                            .temperature
                            .map(|temp| GenerationConfig { temperature: temp });

                        let request_body = GenerateContentRequest {
                            contents: vec![Content {
                                parts,
                                role: "user".to_string(),
                            }],
                            generation_config,
                        };

                        call_gemini_api(client, api_key, base_url, model_name, &request_body).await
                    }
                    None => {
                        // Handle direct output when no model is specified
                        if unit.mime_type.type_() == mime::TEXT {
                            String::from_utf8(unit.data).map_err(|e| {
                                anyhow!("Input '{}' is not valid UTF-8: {}", base_identifier, e)
                            })
                        } else {
                            Err(anyhow!(
                                "Cannot output binary input content ({}) for '{}' when no model is specified.",
                                unit.mime_type,
                                base_identifier
                            ))
                        }
                    }
                };
                match res {
                    Ok(content) => Ok((identifier, content)),
                    Err(e) => Err((identifier, e)),
                }
            }
            WorkItem::ProcessPrompt => {
                let identifier = format!("Prompt (run {}/{})", run_idx + 1, total_runs);
                let model_name = self
                    .model_name
                    .as_ref()
                    .expect("Model name required for ProcessPrompt");
                let prompt = self
                    .prompt
                    .as_ref()
                    .expect("Prompt required for ProcessPrompt");

                let client = self
                    .client
                    .as_ref()
                    .expect("API client should be initialized");
                let api_key = self
                    .api_key
                    .as_ref()
                    .expect("API key should be initialized");
                let base_url = self
                    .base_url
                    .as_ref()
                    .expect("Base URL should be initialized");

                let generation_config = self
                    .temperature
                    .map(|temp| GenerationConfig { temperature: temp });

                let request_body = GenerateContentRequest {
                    contents: vec![Content {
                        parts: vec![Part {
                            text: Some(prompt.clone()),
                            inline_data: None,
                        }],
                        role: "user".to_string(),
                    }],
                    generation_config,
                };

                let res =
                    call_gemini_api(client, api_key, base_url, model_name, &request_body).await;
                match res {
                    Ok(content) => Ok((identifier, content)),
                    Err(e) => Err((identifier, e)),
                }
            }
        }
    }
}
// Removed section separator comment

/// Gathers and prepares all input units from files and URLs concurrently.
async fn gather_input_units(
    inputs: &[String],
    client: &ReqwestClient,
    split_pdf: bool,
    concurrency: usize, // Added concurrency parameter
) -> Result<Vec<InputUnit>> {
    // Use a stream to process inputs concurrently
    let units_stream = stream::iter(inputs)
        .map(|input_str| {
            // Clone necessary items for the async operation
            let client_clone = client.clone(); // ReqwestClient is cheap to clone (Arc internally)
            let input_str_owned = input_str.to_string(); // Own the string

            async move {
                // Call the processing function
                match process_input_string(&input_str_owned, &client_clone, split_pdf).await {
                    Ok(units) => Ok(units), // Keep the Vec<InputUnit>
                    Err(e) => {
                        // Log the error but return Ok(empty) to allow others to proceed
                        eprintln!("Error processing input '{}': {}", input_str_owned, e);
                        Ok(Vec::new())
                    }
                }
            }
        })
        .buffer_unordered(concurrency); // Process concurrently based on the provided limit

    // Collect all the Vec<InputUnit> results and flatten them
    let all_results: Vec<Result<Vec<InputUnit>>> = units_stream.collect().await;

    // Flatten the results, discarding errors after logging them above
    let all_units: Vec<InputUnit> = all_results
        .into_iter()
        .filter_map(Result::ok) // Keep only Ok results
        .flatten() // Flatten Vec<Vec<InputUnit>> into Vec<InputUnit>
        .collect();

    Ok(all_units)
}

async fn call_gemini_api(
    client: &ReqwestClient,
    api_key: &str,
    base_url: &str,
    model_name: &str,
    request_body: &GenerateContentRequest,
) -> Result<String> {
    let url = format!(
        "{}/v1beta/models/{}:generateContent?key={}",
        base_url, model_name, api_key
    );

    let response = client
        .post(&url)
        .header("Content-Type", "application/json")
        .json(request_body)
        .send()
        .await
        .context("Failed to send request to Gemini API")?;

    if response.status().is_success() {
        let response_body: GenerateContentResponse = response
            .json()
            .await
            .context("Failed to parse Gemini API response")?;

        let text_output = response_body
            .candidates
            .unwrap_or_default()
            .into_iter()
            .filter_map(|c| c.content)
            .filter_map(|content| content.parts)
            .flatten()
            .filter_map(|part| part.text)
            .collect::<Vec<String>>()
            .join("");

        Ok(text_output)
    } else {
        let status = response.status();
        let error_text = response
            .text()
            .await
            .unwrap_or_else(|_| "Failed to read error body".to_string());
        bail!(
            "Gemini API request failed with status {}: {}",
            status,
            error_text
        )
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    let http_client = ReqwestClient::new();

    dotenvy::dotenv().ok();

    let api_key_var = "GEMINI_API_KEY";
    let endpoint_override_var = "GEMINI_API_ENDPOINT_OVERRIDE";

    let final_model_name: Option<String> = args.model.map(|m| {
        MODEL_ALIASES
            .get(m.as_str())
            .map(|&resolved| resolved.to_string())
            .unwrap_or(m)
    });

    let mut http_client_arc: Option<Arc<ReqwestClient>> = None;
    let mut api_key_arc: Option<Arc<String>> = None;
    let mut base_url_arc: Option<Arc<String>> = None;

    if final_model_name.is_some() {
        let api_key_res = env::var(api_key_var);
        let endpoint_override = env::var(endpoint_override_var).ok();

        let api_key = api_key_res.context(format!(
            "{} environment variable not found (required when model is specified).",
            api_key_var
        ))?;

        let base_url = endpoint_override
            .clone()
            .unwrap_or_else(|| "https://generativelanguage.googleapis.com".to_string());

        let http_client = ReqwestClient::new();
        http_client_arc = Some(Arc::new(http_client));
        api_key_arc = Some(Arc::new(api_key));
        base_url_arc = Some(Arc::new(base_url));

        eprintln!(
            "Gemini API Client configured. Target base URL: {}",
            base_url_arc.as_ref().unwrap()
        );
    } else {
        eprintln!("No model specified. Will output file content directly.");
    }

    let work_items: Vec<WorkItem> = if args.inputs.is_empty() {
        match (&args.prompt, &final_model_name) {
            (Some(_), Some(_)) => {
                eprintln!(
                    "Preparing prompt-only processing for {} repetitions...",
                    args.repeats
                );
                (0..args.repeats).map(|_| WorkItem::ProcessPrompt).collect()
            }
            _ => {
                eprintln!(
                    "Error: --prompt and --model are required when no input files/URLs are provided."
                );
                return Ok(());
            }
        }
    } else {
        eprintln!("Gathering input units...");
        let input_units =
            gather_input_units(&args.inputs, &http_client, args.split_pdf, args.concurrency)
                .await?;
        if input_units.is_empty() {
            eprintln!("No valid inputs found or read from the provided list.");
            return Ok(());
        }
        eprintln!(
            "Preparing processing for {} input units, each repeated {} times...",
            input_units.len(),
            args.repeats
        );
        input_units
            .into_iter()
            .flat_map(|unit| (0..args.repeats).map(move |_| WorkItem::ProcessInput(unit.clone())))
            .collect()
    };

    if work_items.is_empty() {
        eprintln!("No work items to process.");
        return Ok(());
    }

    let output_mutex = Arc::new(Mutex::new(stdout()));
    let has_errors = Arc::new(AtomicBool::new(false)); // Flag to track errors

    match (&final_model_name, &args.prompt) {
        (Some(model), Some(prompt)) => println!(
            "Below are collated results for model '{}' using prompt '{}'",
            model, prompt
        ),
        (Some(model), None) => println!(
            "Below are collated results for model '{}' (no prompt provided)",
            model
        ),
        (None, _) => println!("Below is the collated file content:"),
    }

    // Create the processor instance with the shared context
    let processor = WorkItemProcessor::new(
        http_client_arc,  // Renamed from client_opt
        api_key_arc,      // Renamed from api_key_opt
        base_url_arc,     // Renamed from base_url_opt
        final_model_name, // Renamed from current_model_name
        args.prompt,
        args.temperature,
    );

    let total_runs = args.repeats;
    let processing_stream = stream::iter(work_items.into_iter().enumerate()) // Iterate directly
        .map(|(idx, work_item)| {
            // Get index and work_item
            let run_idx = idx % total_runs;
            let processor_clone = processor.clone(); // Clone processor for the task
            let output = Arc::clone(&output_mutex);
            let errors_flag_clone = Arc::clone(&has_errors);

            tokio::spawn(async move {
                // Call process on the cloned processor instance
                let result: Result<(String, String), (String, anyhow::Error)> = processor_clone
                    .process(work_item, run_idx, total_runs)
                    .await;

                // The result already contains the identifier in both Ok and Err variants.
                (result, output, errors_flag_clone) // Return the result, output handle, and error flag
            })
        })
        .buffer_unordered(args.concurrency);

    processing_stream
        .for_each(|res| async {
            match res {
                // res is JoinResult<(Result<(String, String), (String, anyhow::Error)>, Arc<Mutex<Stdout>>, Arc<AtomicBool>)>
                // Task completed successfully, and processing within the task succeeded
                Ok((Ok((output_id, content)), output, _errors_flag_clone)) => {
                    let mut locked_stdout = output.lock().await;
                    let output_string = format!(
                        "\n--- START OF: {} ---\n{}\n--- END OF: {} ---\n",
                        output_id,
                        content
                            .lines()
                            .map(|line| format!("| {}", line))
                            .collect::<Vec<String>>()
                            .join("\n"),
                        output_id
                    );
                    if let Err(e) = locked_stdout.write_all(output_string.as_bytes()).await {
                        eprintln!("Error writing to stdout: {}", e);
                    }
                }
                // Task completed successfully, but processing within the task failed
                Ok((Err((identifier, e)), _output, errors_flag)) => {
                    // Use the identifier returned in the Err variant
                    eprintln!("Error processing '{}': {}", identifier, e);
                    errors_flag.store(true, Ordering::SeqCst);
                }
                // Task itself failed to complete (JoinError)
                Err(e) => {
                    eprintln!("Error in processing task join handle: {}", e);
                    // Cannot reliably get the identifier here, but we need to flag that *an* error occurred.
                    // The main `has_errors` flag needs to be accessed. This requires passing it back out
                    // or setting it directly if we had access. Let's set it via the clone we have,
                    // although this feels a bit indirect. A channel might be cleaner for reporting join errors.
                    // For now, let's rely on the final check after the loop, as JoinErrors should be rare.
                    // TODO: Consider setting the flag directly here if possible or using a channel.
                    // but let's rely on the check after the loop for now.
                }
            }
        })
        .await;

    // Check if any errors occurred during processing
    if has_errors.load(Ordering::SeqCst) {
        eprintln!("\nOne or more errors occurred during processing.");
        // Exit with a non-zero status code to indicate failure
        std::process::exit(1);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::{Builder as TempFileBuilder, NamedTempFile};

    #[tokio::test]
    async fn test_process_file_input_text_file() {
        let mut file = TempFileBuilder::new()
            .suffix(".txt")
            .tempfile()
            .expect("Failed to create temp file with .txt suffix");
        let file_content = "Hello, world!";
        writeln!(file, "{}", file_content).expect("Failed to write to temp file");
        file.flush().expect("Failed to flush temp file");

        let path = file.path().to_path_buf();
        let identifier = "test_text.txt".to_string();

        let result = process_file_input(path.clone(), identifier.clone(), false).await;

        assert!(result.is_ok());
        let units = result.unwrap();
        assert_eq!(units.len(), 1);
        let unit = &units[0];
        assert_eq!(unit.identifier, identifier);
        assert_eq!(unit.data, format!("{}\n", file_content).as_bytes()); // Add newline to expected
        assert_eq!(unit.mime_type.essence_str(), mime::TEXT_PLAIN.essence_str()); // Compare essence_str
    }

    #[tokio::test]
    async fn test_process_file_input_read_error() {
        let non_existent_path = PathBuf::from("this_file_does_not_exist_hopefully.txt");
        let identifier = "non_existent.txt".to_string();

        let result = process_file_input(non_existent_path, identifier, false).await;

        assert!(result.is_ok());
        let units = result.unwrap();
        assert!(units.is_empty());
        // We can't easily assert stderr here, but the function should print an error
    }

    #[tokio::test]
    async fn test_process_file_input_non_pdf_with_split_true() {
        let mut file = TempFileBuilder::new()
            .suffix(".txt")
            .tempfile()
            .expect("Failed to create temp file with .txt suffix");
        let file_content = "Just a text file.";
        writeln!(file, "{}", file_content).expect("Failed to write to temp file");
        file.flush().expect("Failed to flush temp file");

        let path = file.path().to_path_buf();
        let identifier = "not_a_pdf.txt".to_string();

        // Call with split_pdf = true
        let result = process_file_input(path.clone(), identifier.clone(), true).await;

        // Should behave the same as the text file test (no splitting attempted)
        assert!(result.is_ok());
        let units = result.unwrap();
        assert_eq!(units.len(), 1);
        let unit = &units[0];
        assert_eq!(unit.identifier, identifier);
        assert_eq!(unit.data, format!("{}\n", file_content).as_bytes()); // Add newline to expected
        assert_eq!(unit.mime_type.essence_str(), mime::TEXT_PLAIN.essence_str());
    }

    // TODO: Add test for process_file_input with PDF splitting (requires sample PDF data - better as integration?)

    #[tokio::test]
    async fn test_process_input_string_file() {
        let mut file = NamedTempFile::new().expect("Failed to create temp file");
        let file_content = "File content for process_input_string";
        writeln!(file, "{}", file_content).expect("Failed to write to temp file");
        file.flush().expect("Failed to flush temp file");

        let file_path_str = file.path().to_str().expect("Path is not valid UTF-8");
        let client = ReqwestClient::new(); // Needed for the function signature

        let result = process_input_string(file_path_str, &client, false).await;

        assert!(result.is_ok());
        let units = result.unwrap();
        assert_eq!(units.len(), 1);
        assert_eq!(units[0].data, format!("{}\n", file_content).as_bytes()); // Add newline to expected
        assert!(
            units[0]
                .identifier
                .contains(file.path().file_name().unwrap().to_str().unwrap())
        );
    }

    #[tokio::test]
    async fn test_process_input_string_url() {
        // --- Mock Server Setup ---
        let mock_server = wiremock::MockServer::start().await;
        let mock_url = mock_server.uri();
        let mock_path = "/test_data.txt";
        let full_url = format!("{}{}", mock_url, mock_path);
        let mock_content = "Content from mock URL";
        let mock_mime = "text/plain";

        wiremock::Mock::given(wiremock::matchers::method("GET"))
            .and(wiremock::matchers::path(mock_path))
            .respond_with(
                wiremock::ResponseTemplate::new(200)
                    .set_body_string(mock_content)
                    .insert_header("Content-Type", mock_mime),
            )
            .mount(&mock_server)
            .await;

        // --- Test Execution ---
        let client = ReqwestClient::new(); // Real client, but directed to mock server
        let result = process_input_string(&full_url, &client, false).await;

        // --- Assertions ---
        assert!(result.is_ok());
        let units = result.unwrap();
        assert_eq!(units.len(), 1);
        let unit = &units[0];
        assert_eq!(unit.identifier, full_url);
        assert_eq!(unit.data, mock_content.as_bytes());
        assert_eq!(unit.mime_type.essence_str(), mime::TEXT_PLAIN.essence_str());
    }

    #[test]
    fn test_model_aliases() {
        assert_eq!(
            MODEL_ALIASES.get("flash-think"),
            Some(&"gemini-2.0-flash-thinking-exp-01-21")
        );
        assert_eq!(MODEL_ALIASES.get("pro"), Some(&"gemini-2.5-pro-exp-03-25"));
        assert_eq!(MODEL_ALIASES.get("flash"), Some(&"gemini-2.0-flash"));
        assert_eq!(MODEL_ALIASES.get("non-existent-alias"), None);
        // Check that an unaliased model name isn't present unless explicitly added
        assert_eq!(MODEL_ALIASES.get("gemini-pro"), None);
    }

    #[tokio::test]
    async fn test_call_gemini_api_success() {
        // --- Mock Server Setup ---
        let mock_server = wiremock::MockServer::start().await;
        let model_name = "mock-model-success";
        let api_key = "mock-api-key";
        let base_url = mock_server.uri();
        let expected_response_text = "Mock API Success Response";

        let mock_response_body = serde_json::json!({
          "candidates": [{
            "content": {
              "parts": [{"text": expected_response_text}],
              "role": "model"
            }
          }]
        });

        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .and(wiremock::matchers::path(format!(
                "/v1beta/models/{}:generateContent",
                model_name
            )))
            .and(wiremock::matchers::query_param("key", api_key))
            .and(wiremock::matchers::header(
                "Content-Type",
                "application/json",
            ))
            // Basic body check - could be more specific if needed
            .and(wiremock::matchers::body_partial_json(
                serde_json::json!({"contents": [{}]}),
            ))
            .respond_with(wiremock::ResponseTemplate::new(200).set_body_json(mock_response_body))
            .mount(&mock_server)
            .await;

        // --- Test Execution ---
        let client = ReqwestClient::new();
        let request_body = GenerateContentRequest {
            contents: vec![Content {
                parts: vec![Part {
                    text: Some("test".to_string()),
                    inline_data: None,
                }],
                role: "user".to_string(),
            }],
            generation_config: None,
        };

        let result = call_gemini_api(&client, api_key, &base_url, model_name, &request_body).await;

        // --- Assertions ---
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), expected_response_text);
    }

    #[tokio::test]
    async fn test_call_gemini_api_failure() {
        // --- Mock Server Setup ---
        let mock_server = wiremock::MockServer::start().await;
        let model_name = "mock-model-fail";
        let api_key = "mock-api-key-fail";
        let base_url = mock_server.uri();
        let error_message = "Invalid API key";

        let mock_error_body = serde_json::json!({
            "error": {
                "code": 400,
                "message": error_message,
                "status": "INVALID_ARGUMENT"
            }
        });

        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .and(wiremock::matchers::path(format!(
                "/v1beta/models/{}:generateContent",
                model_name
            )))
            .and(wiremock::matchers::query_param("key", api_key))
            .respond_with(wiremock::ResponseTemplate::new(400).set_body_json(mock_error_body))
            .mount(&mock_server)
            .await;

        // --- Test Execution ---
        let client = ReqwestClient::new();
        let request_body = GenerateContentRequest {
            contents: vec![Content {
                parts: vec![Part {
                    text: Some("test".to_string()),
                    inline_data: None,
                }],
                role: "user".to_string(),
            }],
            generation_config: None,
        };

        let result = call_gemini_api(&client, api_key, &base_url, model_name, &request_body).await;

        // --- Assertions ---
        assert!(result.is_err());
        let error = result.unwrap_err();
        // Check that the error message contains the status code and the message from the mock response
        assert!(error.to_string().contains("400 Bad Request"));
        assert!(error.to_string().contains(error_message));
    }

    #[tokio::test]
    async fn test_gather_input_units_files_only() {
        let mut file1 = NamedTempFile::new().unwrap();
        writeln!(file1, "content1").unwrap();
        file1.flush().unwrap();
        let path1_str = file1.path().to_str().unwrap().to_string();

        let mut file2 = NamedTempFile::new().unwrap();
        writeln!(file2, "content2").unwrap();
        file2.flush().unwrap();
        let path2_str = file2.path().to_str().unwrap().to_string();

        let inputs = vec![path1_str.clone(), path2_str.clone()];
        let client = ReqwestClient::new();

        let result = gather_input_units(&inputs, &client, false, 5).await; // Added concurrency (default 5)

        assert!(result.is_ok());
        let units = result.unwrap();
        assert_eq!(units.len(), 2);
        assert!(units.iter().any(|u| {
            u.identifier
                .contains(file1.path().file_name().unwrap().to_str().unwrap())
                && u.data == b"content1\n"
        })); // Add newline
        assert!(units.iter().any(|u| {
            u.identifier
                .contains(file2.path().file_name().unwrap().to_str().unwrap())
                && u.data == b"content2\n"
        })); // Add newline
    }

    #[tokio::test]
    async fn test_gather_input_units_urls_only() {
        let mock_server = wiremock::MockServer::start().await;
        let url1 = format!("{}/url1", mock_server.uri());
        let url2 = format!("{}/url2", mock_server.uri());

        wiremock::Mock::given(wiremock::matchers::method("GET"))
            .and(wiremock::matchers::path("/url1"))
            .respond_with(wiremock::ResponseTemplate::new(200).set_body_string("content_url1"))
            .mount(&mock_server)
            .await;
        wiremock::Mock::given(wiremock::matchers::method("GET"))
            .and(wiremock::matchers::path("/url2"))
            .respond_with(wiremock::ResponseTemplate::new(200).set_body_string("content_url2"))
            .mount(&mock_server)
            .await;

        let inputs = vec![url1.clone(), url2.clone()];
        let client = ReqwestClient::new();

        let result = gather_input_units(&inputs, &client, false, 5).await; // Added concurrency (default 5)

        assert!(result.is_ok());
        let units = result.unwrap();
        assert_eq!(units.len(), 2);
        assert!(
            units
                .iter()
                .any(|u| u.identifier == url1 && u.data == b"content_url1")
        );
        assert!(
            units
                .iter()
                .any(|u| u.identifier == url2 && u.data == b"content_url2")
        );
    }

    #[tokio::test]
    async fn test_gather_input_units_mixed() {
        // File setup
        let mut file1 = NamedTempFile::new().unwrap();
        writeln!(file1, "content_file").unwrap();
        file1.flush().unwrap();
        let path1_str = file1.path().to_str().unwrap().to_string();

        // URL setup
        let mock_server = wiremock::MockServer::start().await;
        let url1 = format!("{}/mixed_url", mock_server.uri());
        wiremock::Mock::given(wiremock::matchers::method("GET"))
            .and(wiremock::matchers::path("/mixed_url"))
            .respond_with(wiremock::ResponseTemplate::new(200).set_body_string("content_url_mixed"))
            .mount(&mock_server)
            .await;

        let inputs = vec![path1_str.clone(), url1.clone()];
        let client = ReqwestClient::new();

        let result = gather_input_units(&inputs, &client, false, 5).await; // Added concurrency (default 5)

        assert!(result.is_ok());
        let units = result.unwrap();
        assert_eq!(units.len(), 2);
        assert!(units.iter().any(|u| {
            u.identifier
                .contains(file1.path().file_name().unwrap().to_str().unwrap())
                && u.data == b"content_file\n"
        })); // Add newline
        assert!(
            units
                .iter()
                .any(|u| u.identifier == url1 && u.data == b"content_url_mixed")
        );
    }

    #[tokio::test]
    async fn test_gather_input_units_with_errors() {
        // Valid file setup
        let mut file1 = NamedTempFile::new().unwrap();
        writeln!(file1, "valid_content").unwrap();
        file1.flush().unwrap();
        let path1_str = file1.path().to_str().unwrap().to_string();

        // Invalid file path
        let invalid_path_str = "non_existent_file_gather.txt".to_string();

        // Valid URL setup
        let mock_server = wiremock::MockServer::start().await;
        let url1 = format!("{}/valid_url_gather", mock_server.uri());
        wiremock::Mock::given(wiremock::matchers::method("GET"))
            .and(wiremock::matchers::path("/valid_url_gather"))
            .respond_with(wiremock::ResponseTemplate::new(200).set_body_string("valid_url_content"))
            .mount(&mock_server)
            .await;

        // Invalid URL (will cause connection error or 404 depending on mock setup)
        let invalid_url = format!("{}/invalid_url_gather", mock_server.uri());
        // No mock mounted for invalid_url, so it should fail

        let inputs = vec![
            path1_str.clone(),
            invalid_path_str,
            url1.clone(),
            invalid_url,
        ];
        let client = ReqwestClient::new();

        let result = gather_input_units(&inputs, &client, false, 5).await; // Added concurrency (default 5)

        // Should succeed overall, but only contain units for valid inputs
        assert!(result.is_ok());
        let units = result.unwrap();
        assert_eq!(units.len(), 2); // Only the valid file and valid URL
        assert!(units.iter().any(|u| {
            u.identifier
                .contains(file1.path().file_name().unwrap().to_str().unwrap())
                && u.data == b"valid_content\n"
        })); // Add newline
        assert!(
            units
                .iter()
                .any(|u| u.identifier == url1 && u.data == b"valid_url_content")
        );
        // We can't easily assert stderr here, but errors for the invalid inputs should have been printed
    }

    #[tokio::test]
    async fn test_fetch_url_input_404() {
        // --- Mock Server Setup ---
        let mock_server = wiremock::MockServer::start().await;
        let url_404 = format!("{}/not_found", mock_server.uri());

        // No mock mounted for /not_found, so wiremock returns 404 by default

        // --- Test Execution ---
        let client = ReqwestClient::new();
        let result = fetch_url_input(Url::parse(&url_404).unwrap(), &client).await;

        // --- Assertions ---
        // Should succeed but return Ok(None) because the fetch failed
        assert!(result.is_ok());
        assert!(result.unwrap().is_none());
        // Stderr should contain the 404 error message (cannot assert directly here)
    }

    // TODO: Add tests for fetch_url_input connection error (harder to mock reliably)
    // Argument parsing logic is better covered by integration tests in tests/cli.rs
}
