import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
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
import os
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Log the PORT environment variable
port = int(os.getenv("PORT", 10000))  # Convert to int, default to 10000
logger.info(f"Starting application on port: {port}")

app = FastAPI()

# Debug and set correct path for frontend
frontend_path = os.path.join(os.path.dirname(__file__), "frontend")
logger.debug(f"Frontend path: {frontend_path}, exists: {os.path.exists(frontend_path)}")

# Mount frontend static files at /static
app.mount("/static", StaticFiles(directory=frontend_path, html=True), name="static")

# Serve index.html at the root path
try:
    with open(os.path.join(frontend_path, "index.html")) as f:
        index_html = f.read()
    logger.info("Frontend index.html loaded successfully")
except Exception as e:
    logger.error(f"Failed to load index.html: {str(e)}")
    index_html = "<h1>Error: Frontend not found</h1>"

# Check Chroma database existence
async def initialize_chroma_db():
    CHROMA_PATH = "chroma_1753881695"
    if not os.path.exists(CHROMA_PATH):
        logger.error("Chroma DB does not exist, please pre-build it")
    else:
        logger.info("Chroma DB found, skipping initialization")

# Run Chroma DB check after server startup
@app.on_event("startup")
async def startup_event():
    logger.info("FastAPI startup event triggered")
    asyncio.create_task(initialize_chroma_db())

@app.get("/", response_class=HTMLResponse)
async def read_root():
    return index_html

# Allow CORS for frontend
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

@app.post("/api/analyze")
async def analyze_url(input: URLInput):
    try:
        html, load_time, robots_blocked, mobile_optimized, screenshot = await fetch_html(input.url)
        if not html:
            logger.error("Failed to fetch HTML")
            raise HTTPException(status_code=500, detail="Failed to fetch URL")
        
        logger.debug(f"HTML fetched successfully, length: {len(html)}")
        
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
    except Exception as e:
        logger.error(f"Error in analyze_url: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/api/chat")
async def chat_with_bot(input: ChatInput):
    query = input.query
    scores = input.scores or {}
    try:
        embedding_function = AzureOpenAIEmbeddings(
            azure_deployment=os.getenv("AZURE_OPENAI_API_EMBEDDING_DEPLOYMENT_NAME"),
            openai_api_version=os.getenv("OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_KEY")
        ) if all(os.getenv(k) for k in ["AZURE_OPENAI_API_EMBEDDING_DEPLOYMENT_NAME", "OPENAI_API_VERSION", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_KEY"]) else None

        if not embedding_function:
            if "readability" in query.lower() and scores.get("Readability Score"):
                return {"response": f"The readability score for your website is {scores['Readability Score']}%. This score reflects how easily your content is understood. Would you like tips to improve it?"}
            return {"response": "I'm sorry, I need Azure OpenAI credentials to process your request fully. Please provide a query related to readability for a basic response."}

        CHROMA_PATH = "chroma_1753881695"
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
        docs = db.similarity_search(query, k=3)
        if not docs:
            return {"response": "No relevant information found in the database."}

        context_text = "\n\n---\n\n".join([doc.page_content for doc in docs])
        prompt_template = ChatPromptTemplate.from_template("""
        Answer the question based only on the following context:

        {context}

        Additional data: {scores}

        ---

        Answer the question based on the above context: {question}
        """)
        prompt = prompt_template.format(
            context=context_text,
            scores=f"Readability Score: {scores.get('Readability Score', 'N/A')}%",
            question=query
        )

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

async def fetch_html(url):
    try:
        async with async_playwright() as p:
            browser = await p.firefox.launch(headless=True)
            context = await browser.new_context(
                viewport={"width": 1280, "height": 720},
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) Gecko/20100101 Firefox/115.0"
            )
            page = await context.new_page()
            await page.goto(url, wait_until="domcontentloaded", timeout=80000)
            await page.wait_for_timeout(7000)
            html = await page.content()
            screenshot = await page.screenshot(full_page=True)
            screenshot_b64 = base64.b64encode(screenshot).decode('utf-8')
            load_time = 7.0
            robots_blocked = False
            viewport = await page.evaluate('() => document.querySelector("meta[name=viewport]")?.content')
            await browser.close()
            return html, load_time, robots_blocked, bool(viewport), screenshot_b64
    except PlaywrightError as e:
        logger.error(f"Playwright error fetching URL {url}: {str(e)}")
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
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents][:100]
    return FRE(' '.join(sentences)) / 100

def meta_tag_score(soup):
    title = soup.find("title")
    desc = soup.find("meta", attrs={"name": "description"})
    return (1.0 if title else 0) + (1.0 if desc else 0) / 2

def jsonld_score(html, base_url):
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
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text[:100000])
    return min(len(doc.ents) / len(doc), 1.0) if doc else 0

def paragraph_coherence_score(text):
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
    uvicorn.run(app, host="0.0.0.0", port=port, timeout_keep_alive=60)