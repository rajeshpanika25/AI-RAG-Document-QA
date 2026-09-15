import json
from pypdf import PdfReader

PDF_PATH = "AI_Index_2025_50_Page_Clean.pdf"
OUTPUT_FILE = "document_pages.json"

reader = PdfReader(PDF_PATH)

print("================================")
print("PDF TEXT EXTRACTION")
print("================================")

print("Total pages:", len(reader.pages))

pages = []

for page_number, page in enumerate(reader.pages, start=1):

    text = page.extract_text() or ""

    pages.append({
        "page": page_number,
        "text": text.strip()
    })

    print(
        f"Page {page_number}/{len(reader.pages)} "
        f"- {len(text.strip())} characters"
    )


with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        pages,
        file,
        ensure_ascii=False,
        indent=2
    )


print("\n================================")
print("EXTRACTION COMPLETED")
print("================================")

print("Pages extracted:", len(pages))
print("Saved to:", OUTPUT_FILE)