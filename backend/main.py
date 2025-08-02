import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from playwright.async_api import async_playwright, Error as PlaywrightError
from bs4 import BeautifulSoup
import uvicorn
import spacy
from textstat import flesch_reading_ease as FRE
import extruct
from w3lib.html import get_base_url
import re
from urllib.parse import urljoin, urlparse
from sentence_transformers import SentenceTransformer, util
import numpy as np
import logging
import base64
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
import os
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()

# Load environment variables
load_dotenv()

# Debug and set correct path
print(f"Current directory: {os.getcwd()}")
print(f"Frontend exists: {os.path.exists('frontend')}")  # Relative to main.py
frontend_path = os.path.join(os.path.dirname(__file__), "frontend")
print(f"Adjusted frontend path: {frontend_path}")
print(f"Frontend exists at adjusted path: {os.path.exists(frontend_path)}")

# Mount frontend static files
app.mount("/", StaticFiles(directory=frontend_path, html=True), name="static")

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8001",
        "http://localhost:8002",
        "https://your-app.onrender.com"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class URLInput(BaseModel):
    url: str

class ChatInput(BaseModel):
    query: str
    scores: dict = None

@app.get("/api")
async def root():
    return {"message": "AI Visibility Score API is running. Use POST /api/analyze to analyze a URL or /api/chat for chatbot, or visit /api/docs for API documentation."}

async def fetch_html(url):
    async with async_playwright() as p:
        try:
            logger.debug(f"Launching Firefox browser for URL: {url}")
            browser = await p.firefox.launch(headless=True)
            context = await browser.new_context(
                bypass_csp=True,
                ignore_https_errors=True,
                viewport={"width": 1280, "height": 720},
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/115.0"
                ),
                java_script_enabled=True,
                locale="en-US"
            )
            page = await context.new_page()
            try:
                logger.debug(f"Navigating to {url}")
                start_time = asyncio.get_event_loop().time()
                await page.set_extra_http_headers({
                    "Accept-Language": "en-US,en;q=0.9",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                    "Accept-Encoding": "gzip, deflate, br"
                })
                await page.evaluate("() => { navigator.webdriver = false; }")
                for attempt in range(2):
                    try:
                        await page.goto(url, wait_until="domcontentloaded", timeout=80000)
                        break
                    except PlaywrightError as e:
                        logger.warning(f"Attempt {attempt+1} failed for {url}: {str(e)}")
                        if attempt == 1:
                            raise
                        await page.wait_for_timeout(2000)
                logger.debug(f"Waiting for 7 seconds to allow JavaScript rendering")
                await page.wait_for_timeout(7000)
                html = await page.content()
                logger.debug(f"Captured HTML content, length: {len(html)}")
                screenshot = await page.screenshot(full_page=True)
                screenshot_b64 = base64.b64encode(screenshot).decode('utf-8')
                load_time = asyncio.get_event_loop().time() - start_time
                logger.debug(f"Page load time: {load_time:.2f} seconds")
                robots_url = urljoin(url, "/robots.txt")
                try:
                    logger.debug(f"Checking robots.txt at {robots_url}")
                    robots_response = await page.goto(robots_url, wait_until="domcontentloaded", timeout=10000)
                    robots_content = await robots_response.text() if robots_response else ""
                    robots_blocked = "Disallow: /" in robots_content
                except Exception as e:
                    logger.warning(f"Failed to fetch robots.txt for {url}: {str(e)}")
                    robots_blocked = False
                viewport = await page.evaluate('() => document.querySelector("meta[name=viewport]")?.content')
                logger.debug(f"Viewport meta tag found: {bool(viewport)}")
                await context.close()
                return html, load_time, robots_blocked, bool(viewport), screenshot_b64
            except PlaywrightError as e:
                logger.error(f"Playwright error fetching URL {url}: {str(e)}")
                return None, None, True, False, None
            except Exception as e:
                logger.error(f"Unexpected error fetching URL {url}: {str(e)}")
                return None, None, True, False, None
            finally:
                await browser.close()
        except Exception as e:
            logger.error(f"Failed to launch browser for {url}: {str(e)}")
            return None, None, True, False, None

def extract_visible_text(html, url):
    try:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript", "svg", "iframe", "canvas"]):
            tag.decompose()
        main_content = soup.find("main") or soup.find("article") or soup
        visible_text = main_content.get_text(separator=" ", strip=True)
        total_text = len(html)
        word_count = len(visible_text.split())
        return visible_text, total_text, soup, word_count
    except Exception as e:
        logger.error(f"Error extracting text: {str(e)}")
        return "", len(html), None, 0

def semantic_tags_score(soup):
    if not soup:
        return 0
    semantic_tags = ['article', 'section', 'nav', 'aside', 'header', 'footer', 'main']
    count = sum(1 for tag in semantic_tags if soup.find(tag))
    return min(count / len(semantic_tags), 1.0)

def readability_score(text):
    try:
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 0]
        cleaned_text = ' '.join(sentences[:100])
        fre_score = FRE(cleaned_text)
        sent_lengths = [len(sent.split()) for sent in sentences]
        length_variance = np.var(sent_lengths) if sent_lengths else 0
        length_variance_score = min(length_variance / 100, 1.0)
        return (fre_score / 100 + length_variance_score) / 2
    except Exception as e:
        logger.error(f"Error in readability score: {str(e)}")
        return 0

def meta_tag_score(soup):
    if not soup:
        return 0
    title = soup.find("title")
    desc = soup.find("meta", attrs={"name": "description"})
    og_title = soup.find("meta", property="og:title")
    og_desc = soup.find("meta", property="og:description")
    twitter_card = soup.find("meta", attrs={"name": "twitter:card"})
    title_score = 1.0 if title and 10 <= len(title.text.strip()) <= 70 else 0.5 if title else 0
    desc_score = 1.0 if desc and 50 <= len(desc.get("content", "").strip()) <= 160 else 0.5 if desc else 0
    return (title_score + desc_score + int(bool(og_title)) + int(bool(og_desc)) + int(bool(twitter_card))) / 5

def jsonld_score(html, base_url):
    try:
        metadata = extruct.extract(html, base_url=base_url, syntaxes=['json-ld'], uniform=True)
        jsonld = metadata.get('json-ld')
        if not jsonld:
            return 0
        schema_types = ['Article', 'Organization', 'WebPage', 'Product']
        score = 0
        for item in jsonld:
            item_type = item.get('@type', '')
            if item_type in schema_types:
                required_fields = ['name', 'description'] if item_type == 'Article' else ['name']
                score += 0.5 if all(field in item for field in required_fields) else 0.25
        return min(score, 1.0)
    except Exception as e:
        logger.error(f"Error in JSON-LD score: {str(e)}")
        return 0

def image_alt_score(soup):
    if not soup:
        return 0
    images = soup.find_all("img")
    if not images:
        return 1.0
    with_alt = sum(1 for img in images if img.get("alt") and len(img.get("alt").strip()) > 5)
    return with_alt / len(images)

def heading_structure_score(soup):
    if not soup:
        return 0, {}
    heading_counts = {f"h{i}": len(soup.find_all(f"h{i}")) for i in range(1, 7)}
    score = 0
    score += 0.3 if heading_counts["h1"] == 1 else (0.15 if heading_counts["h1"] > 1 else 0)
    score += 0.2 if heading_counts["h2"] >= 1 else 0
    score += 0.2 if heading_counts["h3"] >= 1 else 0
    score += 0.15 if heading_counts["h4"] >= 1 else 0
    headings = soup.find_all(re.compile('^h[1-6]$'))
    hierarchy_score = 1.0
    last_level = 0
    for h in headings:
        level = int(h.name[1])
        if last_level and level > last_level + 1:
            hierarchy_score -= 0.2
        last_level = level
    score += hierarchy_score * 0.15
    return min(score, 1.0), heading_counts

def entity_density_score(text):
    try:
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text[:100000])
        entities = [ent.text for ent in doc.ents]
        words = len(doc)
        return min(len(entities) / words, 1.0) if words else 0
    except Exception as e:
        logger.error(f"Error in entity density score: {str(e)}")
        return 0

def paragraph_coherence_score(text):
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        paragraphs = [p.strip() for p in re.split(r'\n{2,}', text) if len(p.strip().split()) > 10]
        if len(paragraphs) < 2:
            return 0
        embeddings = model.encode(paragraphs, convert_to_tensor=True)
        similarities = util.cos_sim(embeddings, embeddings).mean().item()
        return min(similarities, 1.0)
    except Exception as e:
        logger.error(f"Error in paragraph coherence score: {str(e)}")
        return 0

def visibility_score(text, total_chars):
    return min(len(text) / total_chars, 1.0) if total_chars else 0

def internal_link_score(soup, base_url):
    if not soup:
        return 0
    domain = urlparse(base_url).netloc
    links = soup.find_all("a", href=True)
    internal_links = [link for link in links if urlparse(urljoin(base_url, link['href'])).netloc == domain]
    if not links:
        return 0
    quality_links = sum(1 for link in internal_links if link.text.strip() and len(link.text.strip()) > 5)
    return min(quality_links / len(links), 1.0)

def content_length_score(word_count):
    if word_count < 300:
        return word_count / 300
    elif word_count > 2000:
        return max(1.0 - (word_count - 2000) / 2000, 0.5)
    return 1.0

def crawlability_score(load_time, robots_blocked):
    load_score = max(1.0 - load_time / 10, 0.5) if load_time is not None else 0
    robots_score = 0 if robots_blocked else 1.0
    return (load_score + robots_score) / 2

@app.post("/api/analyze")
async def analyze_url(input: URLInput):
    html, load_time, robots_blocked, mobile_optimized, screenshot = await fetch_html(input.url)
    if not html:
        return {"error": "Failed to fetch URL"}
    
    visible_text, total_text, soup, word_count = extract_visible_text(html, input.url)
    
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
        "Final AI Visibility Score": round(final_score, 2)
    }

    return {"scores": scores, "screenshot": screenshot}

@app.post("/api/chat")
async def chat_with_bot(input: ChatInput):
    query = input.query
    scores = input.scores or {}

    try:
        # Prepare embedding function
        embedding_function = AzureOpenAIEmbeddings(
            azure_deployment=os.getenv("AZURE_OPENAI_API_EMBEDDING_DEPLOYMENT_NAME"),
            openai_api_version=os.getenv("OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_KEY")
        ) if all(os.getenv(k) for k in ["AZURE_OPENAI_API_EMBEDDING_DEPLOYMENT_NAME", "OPENAI_API_VERSION", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_KEY"]) else None

        if not embedding_function:
            # Fallback to simple response if Azure credentials are missing
            if "readability" in query.lower() and scores.get("Readability Score"):
                return {"response": f"The readability score for your website is {scores['Readability Score']}%. This score reflects how easily your content is understood. Would you like tips to improve it?"}
            return {"response": "I'm sorry, I need Azure OpenAI credentials to process your request fully. Please provide a query related to readability for a basic response."}

        # Load Chroma DB
        CHROMA_PATH = os.getenv("STORAGE_PATH", "../chroma_1753881695")
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

        # Search the DB
        docs = db.similarity_search(query, k=3)
        if not docs:
            return {"response": "No relevant information found in the database."}

        context_text = "\n\n---\n\n".join([doc.page_content for doc in docs])
        prompt_template = ChatPromptTemplate.from_template("""
        Answer the question based only on the following context:

        {context}

        Additional data: Readability Score is {readability_score}

        ---

        Answer the question based on the above context: {question}
        """)
        prompt = prompt_template.format(context=context_text, question=query, readability_score=scores.get("Readability Score", "N/A"))

        # Use Azure ChatOpenAI
        model = AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            openai_api_version=os.getenv("OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_KEY")
        )
        messages = [HumanMessage(content=prompt)]
        response = model.invoke(messages)
        return {"response": response.content}

    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return {"response": f"Error occurred: {str(e)}"}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
