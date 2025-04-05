use anyhow::{anyhow, Context, Result};
use lopdf::{Dictionary, Document, Object};

/// Extracts a single page from a PDF document by copying the document
/// and removing all other pages from the page tree.
///
/// Args:
///     pdf_data: Raw byte data of the input PDF.
///     page_number: The 1-based page number to extract.
///
/// Returns:
///     A Result containing the raw byte data of the single-page PDF,
///     or an error if extraction fails.
pub fn extract_page_from_pdf(pdf_data: &[u8], page_number: u32) -> Result<Vec<u8>> {
    let mut doc =
        Document::load_mem(pdf_data).context("Failed to load PDF data for page extraction")?;

    let original_pages = doc.get_pages();
    let page_count = original_pages.len();
    if page_count == 0 {
        return Err(anyhow!("PDF contains no pages"));
    }

    let target_page_id = *original_pages
        .get(&page_number)
        .ok_or_else(|| {
            anyhow!(
                "Page number {} out of range (1-{})",
                page_number,
                page_count
            )
        })?;

    let catalog_id = doc
        .trailer
        .get(b"Root")
        .and_then(|obj| obj.as_reference())
        .context("Failed to get Root object ID from trailer")?;

    let new_pages_dict = Dictionary::from_iter(vec![
        (b"Type".to_vec(), Object::Name(b"Pages".to_vec())),
        (b"Count".to_vec(), Object::Integer(1)),
        (
            b"Kids".to_vec(),
            Object::Array(vec![Object::Reference(target_page_id)]),
        ),
    ]);

    let new_pages_id = doc.add_object(new_pages_dict);

    {
        let catalog_dict = doc
            .get_object_mut(catalog_id)?
            .as_dict_mut()
            .context("Failed to get Catalog dictionary as mutable for update")?;
        catalog_dict.set(b"Pages".to_vec(), Object::Reference(new_pages_id));
    }

    // Explicitly delete the objects for the pages we are removing
    for (num, page_object_id) in original_pages.iter() {
        if *num != page_number {
            doc.delete_object(*page_object_id);
        }
    }

    // Remove objects that are no longer referenced
    doc.prune_objects();

    // compact_objects() does not seem to exist in this version.
    // Pruning might be the best we can do to remove unreachable objects.

    doc.compress();

    let mut buffer = Vec::new();
    doc.save_to(&mut buffer)
        .context("Failed to save modified single-page PDF to buffer")?;
    Ok(buffer)
}

#[cfg(test)]
mod tests {
    use super::*;
    use lopdf::Object::Reference;

    // Helper to create a dummy multi-page PDF for testing
    fn create_dummy_pdf(num_pages: u32) -> Vec<u8> {
        let mut doc = Document::with_version("1.5");
        let mut page_ids = vec![];

        for i in 1..=num_pages {
            let content_data = format!("/F1 12 Tf BT 72 720 Td (Page {}) Tj ET", i);
            let resources = Dictionary::from_iter(vec![(
                b"Font".to_vec(),
                Object::Dictionary(Dictionary::from_iter(vec![(
                    b"F1".to_vec(),
                    Object::Dictionary(Dictionary::from_iter(vec![
                        (b"Type".to_vec(), Object::Name(b"Font".to_vec())),
                        (b"Subtype".to_vec(), Object::Name(b"Type1".to_vec())),
                        (b"BaseFont".to_vec(), Object::Name(b"Helvetica".to_vec())),
                    ])),
                )])),
            )]);
            let resources_id = doc.add_object(resources);
            let content_id = doc.add_object(lopdf::Stream::new(Dictionary::new(), content_data.into_bytes()));

            let page = Dictionary::from_iter(vec![
                (b"Type".to_vec(), Object::Name(b"Page".to_vec())),
                (b"MediaBox".to_vec(), Object::Array(vec![0.into(), 0.into(), 612.into(), 792.into()])),
                (b"Contents".to_vec(), Reference(content_id)),
                (b"Resources".to_vec(), Reference(resources_id)),
            ]);
            page_ids.push(doc.add_object(page));
        }

        let pages_dict = Dictionary::from_iter(vec![
             (b"Type".to_vec(), Object::Name(b"Pages".to_vec())),
             (b"Count".to_vec(), Object::Integer(num_pages as i64)),
             (b"Kids".to_vec(), Object::Array(page_ids.iter().map(|id| Reference(*id)).collect())),
        ]);
        let pages_id = doc.add_object(pages_dict);

        let catalog_dict = Dictionary::from_iter(vec![
            (b"Type".to_vec(), Object::Name(b"Catalog".to_vec())),
            (b"Pages".to_vec(), Reference(pages_id)),
        ]);
        let catalog_id = doc.add_object(catalog_dict);

        doc.trailer.set("Root", Reference(catalog_id));
        let mut buffer = Vec::new();
        doc.save_to(&mut buffer).unwrap();
        buffer
    }

    #[test]
    fn test_extract_existing_page() {
        let pdf_data = create_dummy_pdf(3);
        let result = extract_page_from_pdf(&pdf_data, 2);
        assert!(result.is_ok());

        let extracted_pdf_data = result.unwrap();
        let extracted_doc = Document::load_mem(&extracted_pdf_data).expect("Extracted PDF should be valid");
        assert_eq!(extracted_doc.get_pages().len(), 1, "Extracted PDF should have exactly one page");

        let page_id = *extracted_doc.get_pages().get(&1).unwrap(); // Page numbers reset to 1-based in the new doc
        let page_dict = extracted_doc.get_object(page_id).unwrap().as_dict().unwrap();
        let contents_ref = page_dict.get(b"Contents").unwrap().as_reference().unwrap();
        let contents_stream = extracted_doc.get_object(contents_ref).unwrap().as_stream().unwrap();
        let content_bytes = contents_stream.content.clone();
        let content_string = String::from_utf8(content_bytes).unwrap();
        assert!(content_string.contains("(Page 2)"), "Content should be from original page 2");

    }

    #[test]
    fn test_extract_first_page() {
        let pdf_data = create_dummy_pdf(3);
        let result = extract_page_from_pdf(&pdf_data, 1);
        assert!(result.is_ok());
        let extracted_doc = Document::load_mem(&result.unwrap()).unwrap();
        assert_eq!(extracted_doc.get_pages().len(), 1);
        let page_id = *extracted_doc.get_pages().get(&1).unwrap();
        let page_dict = extracted_doc.get_object(page_id).unwrap().as_dict().unwrap();
        let contents_ref = page_dict.get(b"Contents").unwrap().as_reference().unwrap();
        let contents_stream = extracted_doc.get_object(contents_ref).unwrap().as_stream().unwrap();
        let content_string = String::from_utf8(contents_stream.content.clone()).unwrap();
        assert!(content_string.contains("(Page 1)"));
    }

     #[test]
    fn test_extract_last_page() {
        let pdf_data = create_dummy_pdf(3);
        let result = extract_page_from_pdf(&pdf_data, 3);
        assert!(result.is_ok());
        let extracted_doc = Document::load_mem(&result.unwrap()).unwrap();
        assert_eq!(extracted_doc.get_pages().len(), 1);
        let page_id = *extracted_doc.get_pages().get(&1).unwrap();
        let page_dict = extracted_doc.get_object(page_id).unwrap().as_dict().unwrap();
        let contents_ref = page_dict.get(b"Contents").unwrap().as_reference().unwrap();
        let contents_stream = extracted_doc.get_object(contents_ref).unwrap().as_stream().unwrap();
        let content_string = String::from_utf8(contents_stream.content.clone()).unwrap();
        assert!(content_string.contains("(Page 3)"));
    }


    #[test]
    fn test_extract_page_out_of_bounds() {
        let pdf_data = create_dummy_pdf(3);
        let result = extract_page_from_pdf(&pdf_data, 4);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Page number 4 out of range"));

        let result_zero = extract_page_from_pdf(&pdf_data, 0);
         assert!(result_zero.is_err());
        assert!(result_zero.unwrap_err().to_string().contains("Page number 0 out of range"));
    }

     #[test]
    fn test_extract_from_single_page_pdf() {
        let pdf_data = create_dummy_pdf(1);
        let result = extract_page_from_pdf(&pdf_data, 1);
        assert!(result.is_ok());
        let extracted_doc = Document::load_mem(&result.unwrap()).unwrap();
        assert_eq!(extracted_doc.get_pages().len(), 1);
         // Check content
        let page_id = *extracted_doc.get_pages().get(&1).unwrap();
        let page_dict = extracted_doc.get_object(page_id).unwrap().as_dict().unwrap();
        let contents_ref = page_dict.get(b"Contents").unwrap().as_reference().unwrap();
        let contents_stream = extracted_doc.get_object(contents_ref).unwrap().as_stream().unwrap();
        let content_string = String::from_utf8(contents_stream.content.clone()).unwrap();
        assert!(content_string.contains("(Page 1)"));
    }

    #[test]
    fn test_extract_from_empty_pdf() {
        let mut doc = Document::with_version("1.5");
        let pages_dict = Dictionary::from_iter(vec![
             (b"Type".to_vec(), Object::Name(b"Pages".to_vec())),
             (b"Count".to_vec(), Object::Integer(0)),
             (b"Kids".to_vec(), Object::Array(vec![])),
        ]);
        let pages_id = doc.add_object(pages_dict);

        let catalog_dict = Dictionary::from_iter(vec![
            (b"Type".to_vec(), Object::Name(b"Catalog".to_vec())),
            (b"Pages".to_vec(), Reference(pages_id)),
        ]);
        let catalog_id = doc.add_object(catalog_dict);
        doc.trailer.set("Root", Reference(catalog_id));
        let mut buffer = Vec::new();
        doc.save_to(&mut buffer).unwrap();

        let result = extract_page_from_pdf(&buffer, 1);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("PDF contains no pages"));
    }
}