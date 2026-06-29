import pypdf

reader = pypdf.PdfReader("bosniaandhercegovina2007en.pdf")

for page_num in range(36, 46):
    print(f"\n--- PAGE {page_num} ---")
    text = reader.pages[page_num - 1].extract_text()
    print(text[:1500])  # Print first 1500 chars of each page
