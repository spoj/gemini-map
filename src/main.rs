use anyhow::{anyhow, bail, Context, Result};
use base64::{engine::general_purpose::STANDARD as BASE64_STANDARD, Engine as _};
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
    sync::{Arc, LazyLock},
};
use tokio::{
    fs,
    io::{stdout, AsyncWriteExt},
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

#[derive(Serialize)]
struct GenerateContentRequest {
    contents: Vec<Content>,
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
// Removed section separator comment

/// Gathers and prepares all input units from files and URLs.
async fn gather_input_units(
    inputs: &[String],
    client: &ReqwestClient,
    split_pdf: bool,
) -> Result<Vec<InputUnit>> {
    let mut all_units = Vec::new();

    for input_str in inputs {
        match process_input_string(input_str, client, split_pdf).await {
            Ok(new_units) => all_units.extend(new_units), // Removed pdfium_opt
            Err(e) => eprintln!("Error processing input '{}': {}", input_str, e),
        }
    }
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
                eprintln!("Error: --prompt and --model are required when no input files/URLs are provided.");
                return Ok(());
            }
        }
    } else {
        eprintln!("Gathering input units...");
        let input_units = gather_input_units(&args.inputs, &http_client, args.split_pdf).await?;
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

    let total_runs = args.repeats;
    let tasks_to_process = work_items
        .into_iter()
        .enumerate()
        .map(|(idx, item)| {
            let run_idx = idx % total_runs;
            (item, run_idx, total_runs)
        })
        .collect::<Vec<_>>();

    let processing_stream = stream::iter(tasks_to_process)
        .map(|(work_item, run_idx, total_runs)| {
            let output = Arc::clone(&output_mutex);
            let prompt_opt = args.prompt.clone();
            let current_model_name = final_model_name.clone();

            let client_opt = http_client_arc.clone();
            let api_key_opt = api_key_arc.clone();
            let base_url_opt = base_url_arc.clone();

            tokio::spawn(async move {
                let (result_identifier, result): (String, Result<String>) = match work_item {
                    WorkItem::ProcessInput(unit) => {
                        let base_identifier = unit.identifier.clone();
                        let identifier = format!("{} (run {}/{})", base_identifier, run_idx + 1, total_runs);

                        let res = match current_model_name {
                            Some(model_name) => {
                                let client = client_opt.as_ref().expect("API client should be initialized for model");
                                let api_key = api_key_opt.as_ref().expect("API key should be initialized for model");
                                let base_url = base_url_opt.as_ref().expect("Base URL should be initialized for model");

                                let mut parts = Vec::new();
                                if let Some(prompt) = prompt_opt {
                                    parts.push(Part { text: Some(prompt), inline_data: None });
                                }
                                let inline_data = InlineData {
                                    mime_type: unit.mime_type.to_string(),
                                    data: BASE64_STANDARD.encode(&unit.data),
                                };
                                parts.push(Part { text: None, inline_data: Some(inline_data) });

                                let request_body = GenerateContentRequest {
                                    contents: vec![Content { parts, role: "user".to_string() }],
                                };

                                call_gemini_api(client, api_key, base_url, &model_name, &request_body).await
                            }
                            None => {
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
                        (identifier, res)
                    }
                    WorkItem::ProcessPrompt => {
                        let identifier = format!("Prompt (run {}/{})", run_idx + 1, total_runs);
                        let model_name = current_model_name.expect("Model name required for ProcessPrompt");
                        let prompt = prompt_opt.expect("Prompt required for ProcessPrompt");

                        let client = client_opt.as_ref().expect("API client should be initialized for prompt");
                        let api_key = api_key_opt.as_ref().expect("API key should be initialized for prompt");
                        let base_url = base_url_opt.as_ref().expect("Base URL should be initialized for prompt");

                        let request_body = GenerateContentRequest {
                            contents: vec![Content {
                                parts: vec![Part { text: Some(prompt), inline_data: None }],
                                role: "user".to_string(),
                            }],
                        };

                        let res = call_gemini_api(client, api_key, base_url, &model_name, &request_body).await;
                        (identifier, res)
                    }
                };

                let final_result: Result<(String, String)> = result.map(|content| (result_identifier.clone(), content));

                (result_identifier, final_result, output)
            })
        })
        .buffer_unordered(args.concurrency);

    processing_stream
        .for_each(|res| async {
            match res {
                Ok((_identifier, Ok((output_id, content)), output)) => {
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
                Ok((identifier, Err(e), _)) => {
                    eprintln!("Error processing input '{}': {}", identifier, e);
                }
                Err(e) => {
                    eprintln!("Error in processing task: {}", e);
                }
            }
        })
        .await;

    Ok(())
}
