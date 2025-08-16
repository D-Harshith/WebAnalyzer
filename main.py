import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from playwright.async_api import async_playwright, TimeoutError
from bs4 import BeautifulSoup
import uvicorn
import logging
import base64
import os
import re
import subprocess
import aiohttp
from urllib.parse import urljoin, urlparse
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Get port
port = int(os.getenv("PORT", 10000))
logger.info(f"Starting application on port: {port}")

app = FastAPI()

# Mount frontend static files
frontend_path = os.path.join(os.path.dirname(__file__), "frontend")
app.mount("/static", StaticFiles(directory=frontend_path, html=True), name="static")

# Serve index.html
try:
    with open(os.path.join(frontend_path, "index.html")) as f:
        index_html = f.read()
    logger.info("Frontend index.html loaded")
except Exception as e:
    logger.error(f"Failed to load index.html: {str(e)}")
    index_html = "<h1>Error: Frontend not found</h1>"

# Check Chroma database
async def check_chroma_db():
    CHROMA_PATH = "chroma_1753881695"
    if not os.path.exists(CHROMA_PATH):
        logger.error("Chroma DB not found")
    else:
        logger.info("Chroma DB found")

@app.on_event("startup")
async def startup_event():
    logger.info("FastAPI startup")
    asyncio.create_task(check_chroma_db())

@app.get("/", response_class=HTMLResponse)
async def read_root():
    return index_html

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

class URLInput(BaseModel):
    url: str

class ChatInput(BaseModel):
    query: str
    scores: dict = None

def validate_url(url: str) -> bool:
    """Validate URL format and scheme."""
    regex = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain
        r'localhost|'  # localhost
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # IP
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    return re.match(regex, url) is not None

@app.post("/api/analyze")
async def analyze_url(input: URLInput):
    try:
        logger.info(f"Starting analysis for URL: {input.url}")
        # Validate URL
        if not validate_url(input.url):
            logger.error(f"Invalid URL format: {input.url}")
            raise HTTPException(status_code=400, detail="Invalid URL format")

        # Fetch HTML
        html, load_time, robots_blocked, mobile_optimized, screenshot = await fetch_html(input.url)
        if not html:
            logger.error(f"Failed to fetch HTML for URL: {input.url}")
            raise HTTPException(status_code=500, detail=f"Failed to fetch URL: {input.url}")
        
        logger.info(f"HTML fetched, length: {len(html)}")
        
        # Import heavy dependencies only when needed
        from textstat import flesch_reading_ease as FRE
        import spacy
        import extruct
        from w3lib.html import get_base_url
        from sentence_transformers import SentenceTransformer, util
        import numpy as np

        # Process HTML
        visible_text, total_text, soup, word_count = extract_visible_text(html, input.url)
        logger.info(f"Extracted visible text, word count: {word_count}")
        
        sem_score = semantic_tags_score(soup)
        read_score = readability_score(visible_text)
        meta_score = meta_tag_score(soup)
        jsonld = jsonld_score(html, base_url=input.url)
        img_score = image_alt_score(soup)
        heading_score, heading_counts = heading_structure_score(soup)
        entity_score = entity_density_score(visible_text)
        para_score = paragraph_coherence_score(visible_text)
        vis_score = visibility_score(visible_text, total_text)
        link_score = internal_link_score(soup, input.url)
        content_score = content_length_score(word_count)
        crawl_score = crawlability_score(load_time, robots_blocked)
        mobile_score = 1.0 if mobile_optimized else 0.5

        final_score = (
            sem_score * 0.15 +
            read_score * 0.15 +
            meta_score * 0.10 +
            jsonld * 0.15 +
            img_score * 0.10 +
            heading_score * 0.10 +
            entity_score * 0.10 +
            para_score * 0.10 +
            vis_score * 0.05 +
            link_score * 0.05 +
            content_score * 0.05 +
            crawl_score * 0.05 +
            mobile_score * 0.05
        ) * 100

        scores = {
            "Semantic Score": round(sem_score * 100, 2),
            "Readability Score": round(read_score * 100, 2),
            "Meta Tag Score": round(meta_score * 100, 2),
            "JSON-LD Score": round(jsonld * 100, 2),
            "Image ALT Score": round(img_score * 100, 2),
            "Heading Structure Score": round(heading_score * 100, 2),
            "H1 Count": heading_counts.get("h1", 0),
            "H2 Count": heading_counts.get("h2", 0),
            "H3 Count": heading_counts.get("h3", 0),
            "H4 Count": heading_counts.get("h4", 0),
            "Entity Density Score": round(entity_score * 100, 2),
            "Paragraph Coherence Score": round(para_score * 100, 2),
            "Visibility Score": round(vis_score * 100, 2),
            "Internal Link Score": round(link_score * 100, 2),
            "Content Length Score": round(content_score * 100, 2),
            "Crawlability Score": round(crawl_score * 100, 2),
            "Mobile Optimization Score": round(mobile_score * 100, 2),
            "Final AI Visibility Score": round(final_score, 2),
        }

        return {
            "scores": scores,
            "screenshot": screenshot if screenshot else ""
        }
    except Exception as e:
        logger.error(f"Error in analysis: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

async def fetch_html(url):
    try:
        async with async_playwright() as playwright:
            for attempt in range(3):
                try:
                    logger.info(f"Launching browser for URL: {url}, attempt {attempt + 1}")
                    browser = await playwright.chromium.launch(headless=True, args=['--no-sandbox', '--disable-setuid-sandbox'])
                    context = await browser.new_context(
                        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                        viewport={"width": 1920, "height": 1080}
                    )
                    page = await context.new_page()
                    response = await page.goto(url, wait_until="networkidle", timeout=30000)
                    if response and response.status >= 400:
                        logger.error(f"Failed to load URL, status: {response.status if response else 'No response'}")
                        break
                    await page.wait_for_load_state("networkidle", timeout=30000)
                    html = await page.content()
                    logger.info(f"Capturing screenshot for: {url}")
                    screenshot = await page.screenshot(full_page=True, timeout=30000)
                    screenshot_b64 = base64.b64encode(screenshot).decode('utf-8')
                    load_time = 7.0  # Placeholder; consider measuring actual time
                    robots_blocked = False  # Adjust if robots.txt check is added
                    viewport = await page.evaluate('() => document.querySelector("meta[name=viewport]")?.content')
                    return html, load_time, robots_blocked, bool(viewport), screenshot_b64
                except TimeoutError as e:
                    logger.warning(f"Attempt {attempt + 1} timed out for URL {url}: {str(e)}")
                    if attempt == 2:
                        raise
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                except Exception as e:
                    logger.warning(f"Attempt {attempt + 1} failed for URL {url}: {str(e)}")
                    if attempt == 2:
                        raise
                    await asyncio.sleep(2 ** attempt)
                finally:
                    logger.info(f"Closing browser for URL: {url}")
                    await browser.close()
    except Exception as e:
        logger.error(f"Error fetching URL {url}: {str(e)}", exc_info=True)
        logger.info(f"Falling back to aiohttp for URL: {url}")
        return await fetch_html_fallback(url)

async def fetch_html_fallback(url):
    """Fallback method to fetch HTML using aiohttp."""
    try:
        logger.info(f"Fetching HTML with aiohttp for URL: {url}")
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:115.0) Gecko/20100101 Firefox/115.0",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1"
        }
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=30), headers=headers) as response:
                if response.status >= 400:
                    logger.error(f"Failed to fetch URL {url} with aiohttp, status: {response.status}")
                    return None, None, True, False, None
                html = await response.text()
                logger.info(f"Successfully fetched HTML with aiohttp for URL: {url}, length: {len(html)}")
                return html, 7.0, False, True, None  # No screenshot in fallback
    except Exception as e:
        logger.error(f"Failed to fetch URL {url} with aiohttp: {str(e)}")
        return None, None, True, False, None

def extract_visible_text(html, url):
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    visible_text = soup.get_text(separator=" ", strip=True)
    return visible_text, len(html), soup, len(visible_text.split())

def semantic_tags_score(soup):
    semantic_tags = ['article', 'section', 'nav', 'aside', 'header', 'footer', 'main']
    count = sum(1 for tag in semantic_tags if soup.find(tag))
    return min(count / len(semantic_tags), 1.0)

def readability_score(text):
    from textstat import flesch_reading_ease as FRE
    import spacy
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents][:100]
    return FRE(' '.join(sentences)) / 100

def meta_tag_score(soup):
    title = soup.find("title")
    desc = soup.find("meta", attrs={"name": "description"})
    return (1.0 if title else 0) + (1.0 if desc else 0) / 2

def jsonld_score(html, base_url):
    import extruct
    from w3lib.html import get_base_url
    metadata = extruct.extract(html, base_url=base_url, syntaxes=['json-ld'])
    return 1.0 if metadata.get('json-ld') else 0

def image_alt_score(soup):
    images = soup.find_all("img")
    if not images:
        return 1.0
    with_alt = sum(1 for img in images if img.get("alt"))
    return with_alt / len(images)

def heading_structure_score(soup):
    heading_counts = {f"h{i}": len(soup.find_all(f"h{i}")) for i in range(1, 5)}
    score = 0.5 if heading_counts["h1"] == 1 else 0
    score += 0.5 if heading_counts["h2"] >= 1 else 0
    return score, heading_counts

def entity_density_score(text):
    import spacy
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text[:100000])
    return min(len(doc.ents) / len(doc), 1.0) if doc else 0

def paragraph_coherence_score(text):
    from sentence_transformers import SentenceTransformer, util
    import numpy as np
    model = SentenceTransformer('all-MiniLM-L6-v2')
    paragraphs = [p for p in text.split('\n\n') if len(p.split()) > 10]
    if len(paragraphs) < 2:
        return 0
    embeddings = model.encode(paragraphs)
    similarities = util.cos_sim(embeddings, embeddings).mean().item()
    return min(similarities, 1.0)

def visibility_score(text, total_chars):
    return min(len(text) / total_chars, 1.0) if total_chars else 0

def internal_link_score(soup, base_url):
    domain = urlparse(base_url).netloc
    links = soup.find_all("a", href=True)
    internal_links = [l for l in links if urlparse(urljoin(base_url, l['href'])).netloc == domain]
    return len(internal_links) / len(links) if links else 0

def content_length_score(word_count):
    return min(word_count / 300, 1.0) if word_count < 300 else 1.0

def crawlability_score(load_time, robots_blocked):
    return 0 if robots_blocked else max(1.0 - load_time / 10, 0.5)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=port, timeout_keep_alive=60, workers=1)