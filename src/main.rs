use anyhow::{anyhow, bail, Context, Result};
use base64::{engine::general_purpose::STANDARD as BASE64_STANDARD, Engine as _};
use clap::Parser;
use futures::stream::{self, StreamExt};
use image::{DynamicImage, ImageFormat};
use pdfium_render::prelude::*;
use reqwest::Client as ReqwestClient;
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    env,
    io::Cursor, // Re-added Cursor for image encoding
    path::PathBuf,
    str,
    sync::{Arc, LazyLock},
};
use tokio::{
    fs,
    io::{stdout, AsyncWriteExt},
    sync::Mutex,
};
use url::Url;

// --- API Structs ---
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
    data: String, // Base64 encoded data
}

#[derive(Serialize)]
struct Content {
    parts: Vec<Part>,
    role: String, // Added role field
}

#[derive(Serialize)]
struct GenerateContentRequest {
    contents: Vec<Content>,
    // generationConfig: Option<...>, // Add if needed
    // safetySettings: Option<...>, // Add if needed
}

#[derive(Deserialize, Debug)]
struct GenerateContentResponse {
    candidates: Option<Vec<Candidate>>,
    // promptFeedback: Option<...>,
}

#[derive(Deserialize, Debug)]
struct Candidate {
    content: Option<ContentResponse>,
    // finishReason: Option<String>,
    // index: Option<u32>,
    // safetyRatings: Option<...>,
}

#[derive(Deserialize, Debug)]
struct ContentResponse {
    parts: Option<Vec<PartResponse>>,
    // role: Option<String>, // Usually "model" for responses
}

#[derive(Deserialize, Debug)]
struct PartResponse {
    text: Option<String>,
}

// Represents a single unit of input data to be processed.
// This will allow handling files, URLs, PDF pages etc. uniformly later.
#[derive(Debug, Clone)]
struct InputUnit {
    identifier: String,
    data: Vec<u8>,
    mime_type: mime::Mime,
}
// Represents the type of work to be performed in a concurrent task.
#[derive(Debug, Clone)]
enum WorkItem {
    ProcessInput(InputUnit), // Process data from a file/URL
    ProcessPrompt,           // Process only the prompt (no input data)
}

// --- Model Aliases ---
static MODEL_ALIASES: LazyLock<HashMap<&'static str, &'static str>> = LazyLock::new(|| {
    let mut m = HashMap::new();
    m.insert("flash-think", "gemini-2.0-flash-thinking-exp-01-21");
    m.insert("pro", "gemini-2.5-pro-exp-03-25");
    m.insert("flash", "gemini-2.0-flash");
    // Add other aliases if needed
    m
});

/// Apply a Gemini prompt to multiple files concurrently.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// The prompt to send to the Gemini model along with the file content. Optional.
    #[arg(short, long)]
    prompt: Option<String>, // Made optional

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
    // Removed required = true
    inputs: Vec<String>, // Changed from PathBuf to String to allow URLs later

    /// Split PDF files into individual pages (rendered as PNGs) instead of processing the whole file. Requires PDFium library.
    #[arg(short, long, default_value_t = false)]
    split_pdf: bool,
}

/// Fetches content from a URL and creates an InputUnit.
async fn fetch_url_input(url: Url, client: &ReqwestClient) -> Result<Option<InputUnit>> {
    let identifier = url.to_string();
    println!("Fetching URL: {}", identifier);
    match client.get(url).send().await {
        Ok(response) => {
            if response.status().is_success() {
                // Try to guess MIME from headers first, then URL path
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
                        Ok(None) // Treat as non-fatal for this input
                    }
                }
            } else {
                eprintln!(
                    "Error fetching URL {}: Status {}",
                    identifier,
                    response.status()
                );
                Ok(None) // Treat as non-fatal for this input
            }
        }
        Err(e) => {
            eprintln!("Error sending request to URL {}: {}", identifier, e);
            Ok(None) // Treat as non-fatal for this input
        }
    }
}

/// Renders pages of a PDF document into PNG InputUnits.
fn split_pdf_into_pages(pdfium: &Pdfium, data: &[u8], identifier: &str) -> Result<Vec<InputUnit>> {
    let mut page_units = Vec::new();
    match pdfium.load_pdf_from_byte_slice(data, None) {
        Ok(document) => {
            let page_count = document.pages().len();
            println!("Splitting PDF: {} ({} pages)", identifier, page_count);
            let render_config = PdfRenderConfig::new(); // Default config

            for (index, page) in document.pages().iter().enumerate() {
                let page_num = index + 1;
                match page.render_with_config(&render_config) {
                    Ok(bitmap) => {
                        let dynamic_image: DynamicImage = bitmap.as_image();
                        let mut png_data = Cursor::new(Vec::new());
                        match dynamic_image.write_to(&mut png_data, ImageFormat::Png) {
                            Ok(_) => {
                                let page_identifier = format!("{} (page {})", identifier, page_num);
                                page_units.push(InputUnit {
                                    identifier: page_identifier,
                                    data: png_data.into_inner(),
                                    mime_type: mime::IMAGE_PNG,
                                });
                            }
                            Err(e) => eprintln!(
                                "Error encoding page {} of {} to PNG: {}",
                                page_num, identifier, e
                            ),
                        }
                    }
                    Err(e) => {
                        eprintln!("Error rendering page {} of {}: {}", page_num, identifier, e)
                    }
                }
            }
        }
        Err(e) => eprintln!("Error loading PDF document {}: {}", identifier, e),
    }
    Ok(page_units)
}

/// Reads a file and creates InputUnits, potentially splitting PDFs.
async fn process_file_input(
    path: PathBuf,
    identifier: String,
    pdfium_opt: Option<&Pdfium>,
    split_pdf: bool,
) -> Result<Vec<InputUnit>> {
    match fs::read(&path).await {
        Ok(data) => {
            let mime_type = mime_guess::from_path(&path).first_or(mime::APPLICATION_OCTET_STREAM);

            // Check if it's a PDF and if splitting is requested and possible
            if mime_type == mime::APPLICATION_PDF && split_pdf {
                if let Some(pdfium) = pdfium_opt {
                    // Attempt to split PDF
                    split_pdf_into_pages(pdfium, &data, &identifier)
                } else {
                    // Pdfium not available, process as whole file
                    eprintln!("PDFium library not available. Cannot split PDF '{}'. Processing as whole file.", identifier);
                    Ok(vec![InputUnit {
                        identifier,
                        data,
                        mime_type,
                    }])
                }
            } else {
                // Handle non-PDF files or non-split PDFs
                Ok(vec![InputUnit {
                    identifier,
                    data,
                    mime_type,
                }])
            }
        }
        Err(e) => {
            eprintln!("Error reading file {}: {}", identifier, e);
            Ok(Vec::new()) // Treat as non-fatal for this input
        }
    }
}

/// Processes a single input string (URL or file path).
async fn process_input_string(
    input_str: &str,
    client: &ReqwestClient,
    pdfium_opt: Option<&Pdfium>,
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
            process_file_input(path, identifier, pdfium_opt, split_pdf).await
        }
    }
}

// --- Main Input Gathering Function ---

/// Gathers and prepares all input units from files and URLs.
async fn gather_input_units(
    inputs: &[String],
    client: &ReqwestClient,
    split_pdf: bool,
) -> Result<Vec<InputUnit>> {
    let pdfium_opt = Some(Pdfium::default()); // Initialize Pdfium once
    let mut all_units = Vec::new();

    for input_str in inputs {
        match process_input_string(input_str, client, pdfium_opt.as_ref(), split_pdf).await {
            Ok(new_units) => all_units.extend(new_units),
            Err(e) => eprintln!("Error processing input '{}': {}", input_str, e), // Log fatal errors from processing
        }
    }
    Ok(all_units)
}
// --- Gemini API Call Helper ---

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

        // Extract text from the response
        let text_output = response_body
            .candidates
            .unwrap_or_default()
            .into_iter()
            .filter_map(|c| c.content)
            .filter_map(|content| content.parts)
            .flatten()
            .filter_map(|part| part.text)
            .collect::<Vec<String>>()
            .join(""); // Concatenate text parts

        Ok(text_output)
    } else {
        let status = response.status();
        let error_text = response.text().await.unwrap_or_else(|_| "Failed to read error body".to_string());
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

    // Initialize Reqwest client needed for potential URL downloads
    let http_client = ReqwestClient::new();

    // Structs moved to module level

    // Optional: Initialize tracing/logging
    // tracing_subscriber::fmt::init(); // Example

    // Load .env file if present (useful for development)
    dotenvy::dotenv().ok();

    // --- API Client Initialization ---
    let api_key_var = "GEMINI_API_KEY";
    let endpoint_override_var = "GEMINI_API_ENDPOINT_OVERRIDE";

    // Resolve model name using aliases
    let final_model_name: Option<String> = args.model.map(|m| {
        MODEL_ALIASES
            .get(m.as_str())
            .map(|&resolved| resolved.to_string())
            .unwrap_or(m) // Use original name if no alias found
    });

    // --- API Config (only if model is specified) ---
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

        let base_url = endpoint_override.clone().unwrap_or_else(|| {
            // Clone endpoint_override
            "https://generativelanguage.googleapis.com".to_string()
        });

        let http_client = ReqwestClient::new();
        http_client_arc = Some(Arc::new(http_client));
        api_key_arc = Some(Arc::new(api_key));
        base_url_arc = Some(Arc::new(base_url));

        println!(
            "Gemini API Client configured. Target base URL: {}",
            base_url_arc.as_ref().unwrap() // Safe unwrap due to is_some check
        );
    } else {
        println!("No model specified. Will output file content directly.");
    }

    // --- Determine Work Items ---
    let work_items: Vec<WorkItem> = if args.inputs.is_empty() {
        // Case: No inputs provided
        match (&args.prompt, &final_model_name) {
            (Some(_), Some(_)) => {
                // Prompt and model are present, create prompt-only work items
                println!(
                    "Preparing prompt-only processing for {} repetitions...",
                    args.repeats
                );
                (0..args.repeats).map(|_| WorkItem::ProcessPrompt).collect()
            }
            _ => {
                // Either prompt or model (or both) are missing
                eprintln!("Error: --prompt and --model are required when no input files/URLs are provided.");
                return Ok(()); // Exit early
            }
        }
    } else {
        // Case: Inputs provided
        println!("Gathering input units...");
        let input_units = gather_input_units(&args.inputs, &http_client, args.split_pdf).await?;
        if input_units.is_empty() {
            println!("No valid inputs found or read from the provided list.");
            return Ok(()); // Exit early
        }
        println!("Preparing processing for {} input units, each repeated {} times...", input_units.len(), args.repeats);
        // Create input-based work items, repeated N times
        input_units
            .into_iter()
            .flat_map(|unit| {
                (0..args.repeats).map(move |_| WorkItem::ProcessInput(unit.clone()))
            })
            .collect()
    };
 
    if work_items.is_empty() {
        // This case should ideally be caught earlier, but added as a safeguard
        println!("No work items to process.");
        return Ok(());
    }
 
    // --- Concurrent Processing ---
    let output_mutex = Arc::new(Mutex::new(stdout())); // Wrap stdout in Arc<Mutex>

    // Adjust header based on model and prompt presence
    match (&final_model_name, &args.prompt) {
        (Some(model), Some(prompt)) => println!(
            "Below are collated results for model '{}' using prompt '{}'",
            model, prompt
        ),
        (Some(model), None) => println!(
            "Below are collated results for model '{}' (no prompt provided)",
            model
        ),
        (None, _) => println!("Below is the collated file content:"), // Prompt doesn't matter if no model specified
    }

    // Combine work items with their run index and total runs
    let total_runs = args.repeats; // Total runs is always args.repeats per original item
    let tasks_to_process = work_items
        .into_iter()
        .enumerate() // Add an overall index
        .map(|(idx, item)| {
            let run_idx = idx % total_runs; // Calculate the run number (0-based) for this item
            (item, run_idx, total_runs)
        })
        .collect::<Vec<_>>();
 
    let processing_stream = stream::iter(tasks_to_process)
        .map(|(work_item, run_idx, total_runs)| { // Now iterates over (WorkItem, run_idx, total_runs)
            // Clone necessary data for the async task
            let output = Arc::clone(&output_mutex);
            let prompt_opt = args.prompt.clone();
            let current_model_name = final_model_name.clone();

            // Clone Option<Arc<T>> safely
            let client_opt = http_client_arc.clone();
            let api_key_opt = api_key_arc.clone();
            let base_url_opt = base_url_arc.clone();

            // Spawn a task for each work item/run combination
            tokio::spawn(async move {
                // Determine the identifier and perform the core work based on WorkItem type
                let (result_identifier, result): (String, Result<String>) = match work_item {
                    WorkItem::ProcessInput(unit) => {
                        let base_identifier = unit.identifier.clone();
                        let identifier = format!("{} (run {}/{})", base_identifier, run_idx + 1, total_runs);

                        let res = match current_model_name {
                            // --- Case 1.1: Input + Model ---
                            Some(model_name) => {
                                let client = client_opt.as_ref().expect("API client should be initialized for model");
                                let api_key = api_key_opt.as_ref().expect("API key should be initialized for model");
                                let base_url = base_url_opt.as_ref().expect("Base URL should be initialized for model");

                                let mut parts = Vec::new();
                                // Add prompt text part if present
                                if let Some(prompt) = prompt_opt {
                                    parts.push(Part { text: Some(prompt), inline_data: None });
                                }
                                // Add inline data part
                                let inline_data = InlineData {
                                    mime_type: unit.mime_type.to_string(),
                                    data: BASE64_STANDARD.encode(&unit.data),
                                };
                                parts.push(Part { text: None, inline_data: Some(inline_data) });

                                let request_body = GenerateContentRequest {
                                    contents: vec![Content { parts, role: "user".to_string() }],
                                };

                                // Call the API helper
                                call_gemini_api(client, api_key, base_url, &model_name, &request_body).await
                            }
                            // --- Case 1.2: Input + No Model ---
                            None => {
                                // Output raw text content if possible, otherwise error
                                if unit.mime_type.type_() == mime::TEXT {
                                    String::from_utf8(unit.data)
                                        .map_err(|e| anyhow!("Input '{}' is not valid UTF-8: {}", base_identifier, e))
                                } else {
                                    Err(anyhow!(
                                        "Cannot output binary input content ({}) for '{}' when no model is specified.",
                                        unit.mime_type, base_identifier
                                    ))
                                }
                            }
                        };
                        (identifier, res) // Return the calculated identifier and the result
                    }
                    WorkItem::ProcessPrompt => {
                        let identifier = format!("Prompt (run {}/{})", run_idx + 1, total_runs);
                        // Model name and prompt must exist for this variant (checked during work_item creation)
                        let model_name = current_model_name.expect("Model name required for ProcessPrompt");
                        let prompt = prompt_opt.expect("Prompt required for ProcessPrompt");

                        let client = client_opt.as_ref().expect("API client should be initialized for prompt");
                        let api_key = api_key_opt.as_ref().expect("API key should be initialized for prompt");
                        let base_url = base_url_opt.as_ref().expect("Base URL should be initialized for prompt");

                        // Construct request body with only the prompt
                        let request_body = GenerateContentRequest {
                            contents: vec![Content {
                                parts: vec![Part { text: Some(prompt), inline_data: None }],
                                role: "user".to_string(),
                            }],
                        };

                        // Call the API helper
                        let res = call_gemini_api(client, api_key, base_url, &model_name, &request_body).await;
                        (identifier, res) // Return the calculated identifier and the result
                    }
                };

                // Wrap the inner result (Result<String>) to match the expected tuple structure
                // for the stream consumer: Result<(String, String)> which represents Result<(Identifier, Content)>
                let final_result: Result<(String, String)> = result.map(|content| (result_identifier.clone(), content));

                (result_identifier, final_result, output) // Pass identifier, final result, and output handle
            })
        })
        .buffer_unordered(args.concurrency);

    processing_stream
        .for_each(|res| async {
            match res {
                Ok((_identifier, Ok((output_id, content)), output)) => {
                    // Use output_id (includes run info)
                    let mut locked_stdout = output.lock().await; // Lock stdout for writing
                    let output_string = format!(
                        "\n--- START OF: {} ---\n{}\n--- END OF: {} ---\n",
                        output_id, // Use identifier with run info
                        content // Use the content (API response or raw text)
                            .lines()
                            .map(|line| format!("| {}", line))
                            .collect::<Vec<String>>()
                            .join("\n"),
                        output_id // Use identifier with run info
                    );
                    // Write output block
                    if let Err(e) = locked_stdout.write_all(output_string.as_bytes()).await {
                        eprintln!("Error writing to stdout: {}", e);
                    }
                    // Mutex is unlocked when locked_stdout goes out of scope
                }
                Ok((identifier, Err(e), _)) => {
                    // Error occurred during processing for a specific unit/run
                    eprintln!("Error processing input '{}': {}", identifier, e);
                }
                Err(e) => {
                    // Error occurred during task spawning/joining (less common)
                    eprintln!("Error in processing task: {}", e);
                }
            }
        })
        .await; // Wait for all tasks to complete


    Ok(())
}

// Removed orphaned code from previous incorrect diff application
