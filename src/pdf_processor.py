import pypdf
import re

def extract_raw_text(pdf_path):
    """
    Extracts raw text from a PDF document.
    """
    reader = pypdf.PdfReader(pdf_path)
    full_text = ""
    for idx, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if page_text:
            full_text += page_text + "\n"
    return full_text

def clean_text(text):
    """
    Cleans raw text extracted from PDF, normalizing line breaks and spacing.
    """
    # Replace multiple newlines with single ones
    text = re.sub(r'\n+', '\n', text)
    # Replace multiple spaces with single spaces
    text = re.sub(r'[ \t]+', ' ', text)
    # Fix hyphenated words at line breaks (e.g. de- \n velopment -> development)
    text = re.sub(r'(\w+)-\n\s*(\w+)', r'\1\2', text)
    return text.strip()

def create_chunks(text, chunk_size=800, overlap=150):
    """
    Implements a sliding window chunking strategy to segment text into overlapping chunks.
    """
    chunks = []
    start = 0
    step = chunk_size - overlap
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += step
    return chunks

def extract_chapter_texts(pdf_path, skip_toc_pages=10):
    """
    Dynamically identifies chapter headings and segments the PDF into individual chapter texts.
    Returns a dictionary mapping chapter number (int) to the text content of that chapter.
    """
    reader = pypdf.PdfReader(pdf_path)
    num_pages = len(reader.pages)
    
    # Common pattern for chapter headings
    chapter_pattern = re.compile(r'\bCHAPTER\s+(\d+)\b', re.IGNORECASE)
    
    chapter_starts = {}
    
    # 1. Detect where each chapter starts
    for idx in range(skip_toc_pages, num_pages):
        text = reader.pages[idx].extract_text()
        if not text:
            continue
        match = chapter_pattern.search(text)
        if match:
            ch_num = int(match.group(1))
            if ch_num not in chapter_starts:
                chapter_starts[ch_num] = idx
                
    chapter_keys = sorted(chapter_starts.keys())
    chapter_texts = {}
    
    if not chapter_starts:
        # Fallback if no CHAPTER X patterns detected: return full text split in 10 equal parts
        print("Warning: No chapter markers detected. Falling back to block splitting.")
        full_text = extract_raw_text(pdf_path)
        words = full_text.split()
        num_chapters = 10
        words_per_ch = len(words) // num_chapters
        for i in range(num_chapters):
            start_w = i * words_per_ch
            end_w = (i + 1) * words_per_ch if i < num_chapters - 1 else len(words)
            chapter_texts[i + 1] = " ".join(words[start_w:end_w])
        return chapter_texts

    # 2. Extract text between chapter boundaries
    for i, ch_num in enumerate(chapter_keys):
        start_page = chapter_starts[ch_num]
        
        # End page is the start page of the next detected chapter
        if i + 1 < len(chapter_keys):
            next_ch = chapter_keys[i + 1]
            end_page = chapter_starts[next_ch]
        else:
            # For the last chapter, look for Annexes/Bibliography or end of document
            end_page = num_pages
            # Scan pages in the last chapter to see if we hit Bibliography/Annexes
            for idx in range(start_page, num_pages):
                text_lower = reader.pages[idx].extract_text().lower()
                if "annexes" in text_lower or "bibliography" in text_lower or "annex 1" in text_lower:
                    end_page = idx
                    break
        
        # Extract and compile chapter text
        ch_text = ""
        for idx in range(start_page, end_page):
            pg_text = reader.pages[idx].extract_text()
            if pg_text:
                ch_text += pg_text + "\n"
        
        chapter_texts[ch_num] = clean_text(ch_text)
        
    return chapter_texts

def scan_themes_by_page(pdf_path):
    """
    Scans the PDF page by page and records which of the 7 core themes are present on each page.
    Returns a list of dicts: [{'page': i, 'theme': theme, 'presence': 1}]
    """
    theme_stems = {
        "education": r'\b(school|educat|teach|learn|literac|train|student|class|enroll)\b',
        "health": r'\b(health|medic|doctor|hospital|diseas|illness|clinic|vaccin|mortality|expectancy)\b',
        "inequality": r'\b(inequal|povert|poor|disadvantag|gap|incom|distribut|wealth|exclu)\b',
        "economy": r'\b(econom|gdp|gni|financ|market|trade|industr|growth|fiscal|tax|recession)\b',
        "gender": r'\b(gender|women|men|female|male|equal|girl|boy|bias|discrim)\b',
        "climate": r'\b(climate|environ|carbon|emiss|warm|green|pollut|energy|sus|natur|ecology)\b',
        "employment": r'\b(employ|job|work|labor|labour|unemploy|career|wage|salar|occupation)\b'
    }
    
    reader = pypdf.PdfReader(pdf_path)
    results = []
    
    for idx, page in enumerate(reader.pages):
        page_num = idx + 1
        page_text = page.extract_text()
        if not page_text:
            continue
        page_text_lower = page_text.lower()
        for theme, pattern in theme_stems.items():
            if re.search(pattern, page_text_lower):
                results.append({
                    "page": page_num,
                    "theme": theme,
                    "presence": 1
                })
    return results

