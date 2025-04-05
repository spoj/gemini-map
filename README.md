# gemini-map

A command-line tool to apply a Gemini prompt to multiple files or URLs concurrently.

## Usage

```bash
cargo run --package gemini-map -- \
    -p <prompt> \
    -m <model_name> \
    [-c <concurrency>] \
    [-s | --split-pdf] \
    <file_or_url1> [<file_or_url2> ...]`

# Example:
cargo run --package gemini-map -- \
    -p "Summarize the key points in this document." \
    -m "gemini-pro" \
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

# Example splitting a PDF into pages (requires PDFium library):
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
gemini-map -p <prompt> -m <model> [-s] <files_or_urls...>
```

## Configuration

Requires the `GEMINI_API_KEY` environment variable to be set for authenticating with the Google Gemini API.

```bash
export GEMINI_API_KEY="YOUR_API_KEY_HERE"
```

### PDF Splitting (`--split-pdf`)

Using the `-s` or `--split-pdf` flag requires the **PDFium library** to be installed on your system. This flag renders each page of a PDF input into a PNG image, which is then processed individually. If the flag is not used, or if the PDFium library cannot be initialized, the entire PDF file is processed as a single binary blob (`application/pdf`).

Please refer to the [`pdfium-render` crate documentation](https://crates.io/crates/pdfium-render) for instructions on installing the PDFium library for your operating system.

## Development

This crate includes integration tests that use `wiremock` to mock the Gemini API.

To run tests:

```bash
cargo test --package gemini-map
```

You might need to set the `GEMINI_API_ENDPOINT_OVERRIDE` environment variable for certain tests if they rely on a specific mock server endpoint.