# gemini-map

A command-line tool to apply a Gemini prompt to multiple files or URLs concurrently.

## Usage

```bash
cargo run --package gemini-map -- \
    -p <prompt> \
    -m <model_name> \
    [-c <concurrency>] \
    [-s | --split-pdf] \
    [-t <temperature>] \
    <file_or_url1> [<file_or_url2> ...]`

# Example:
cargo run --package gemini-map -- \
    -p "Summarize the key points in this document." \
    -m "gemini-pro" \
    -t 0.7 \
    report.txt notes.md chapter1.txt

# Example with custom concurrency:
cargo run --package gemini-map -- \
    -p "Extract action items." \
    -m "gemini-pro" \
    -c 10 \
    meeting_minutes/*.log
```

# Example with a URL:
cargo run --package gemini-map -- \
    -p "What is the main topic of this page?" \
    -m "flash" \
    https://example.com/some/webpage.html
```

# Example splitting a PDF into individual pages:
cargo run --package gemini-map -- \
    -p "Describe the content of this page." \
    -m "gemini-pro-vision" \
    -s \
    multi_page_document.pdf
```

## Installation

```bash
# Build and install locally from the workspace root
cargo install --path crates/gemini-map

# Then run the installed binary
gemini-map -p <prompt> -m <model> [-s] [-t <temperature>] <files_or_urls...>
```

## Configuration

Requires the `GEMINI_API_KEY` environment variable to be set for authenticating with the Google Gemini API.

```bash
export GEMINI_API_KEY="YOUR_API_KEY_HERE"
```

### PDF Splitting (`--split-pdf`)

Using the `-s` or `--split-pdf` flag processes each page of a PDF input individually. It uses the `lopdf` crate to extract each page into a separate, single-page PDF document in memory before sending it for processing. If the flag is not used, the entire PDF file is processed as a single binary blob (`application/pdf`). This feature does **not** require any external libraries like PDFium.

## Development

This crate includes integration tests that use `wiremock` to mock the Gemini API.

To run tests:

```bash
cargo test --package gemini-map
```

You might need to set the `GEMINI_API_ENDPOINT_OVERRIDE` environment variable for certain tests if they rely on a specific mock server endpoint.