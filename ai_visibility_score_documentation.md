# AI Visibility Score Documentation

The AI Visibility Score evaluates a website's visibility to AI systems based on multiple metrics. Below are the key components:

## Semantic Score
The Semantic Score measures the use of semantic HTML tags (`<article>`, `<section>`, `<nav>`, `<aside>`, `<header>`, `<footer>`, `<main>`) to structure content. These tags help AI systems understand the page's organization. The score is calculated as the ratio of used semantic tags to the total possible tags (7), capped at 1.0, and contributes 15% to the final score.

## Readability Score
The Readability Score assesses how easily the website's text can be understood using the Flesch Reading Ease formula. It processes up to 100 sentences with SpaCy and normalizes the score to a 0-1 range, contributing 15% to the final score.

## Meta Tag Score
The Meta Tag Score evaluates the presence of `<title>` and `<meta name="description">` tags, which are critical for AI indexing. It awards 1.0 for each present tag, averaged, and contributes 10% to the final score.

## JSON-LD Score
The JSON-LD Score checks for the presence of JSON-LD structured data, which enhances AI understanding of content. It scores 1.0 if present, 0 otherwise, contributing 15% to the final score.

## Image ALT Score
The Image ALT Score measures the percentage of `<img>` tags with an `alt` attribute, aiding accessibility and AI context. It contributes 10% to the final score.

## Heading Structure Score
The Heading Structure Score evaluates the use of `<h1>` (ideally one) and `<h2>` tags for proper content hierarchy. It awards 0.5 for a single `<h1>` and 0.5 for at least one `<h2>`, contributing 10% to the final score.

## Entity Density Score
The Entity Density Score calculates the proportion of named entities (e.g., people, places) in the text using SpaCy, capped at 1.0, contributing 10% to the final score.

## Paragraph Coherence Score
The Paragraph Coherence Score measures semantic similarity between paragraphs using SentenceTransformer embeddings, contributing 10% to the final score.

## Visibility Score
The Visibility Score is the ratio of visible text to total HTML characters, indicating content prominence, contributing 5% to the final score.

## Internal Link Score
The Internal Link Score calculates the proportion of internal links relative to all links, aiding AI navigation, contributing 5% to the final score.

## Content Length Score
The Content Length Score evaluates word count, with a maximum score for 300+ words, contributing 5% to the final score.

## Crawlability Score
The Crawlability Score assesses page load time and robots.txt restrictions, contributing 5% to the final score.

## Mobile Optimization Score
The Mobile Optimization Score checks for a viewport meta tag, scoring 1.0 if present, 0.5 otherwise, contributing 5% to the final score.

## Final AI Visibility Score
The Final AI Visibility Score is a weighted sum of all metrics, multiplied by 100 to yield a percentage.