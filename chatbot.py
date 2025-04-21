import re
import logging
import random
import time
import string
import nltk
import requests
import wikipediaapi
from typing import List, Dict, Tuple, Optional, Union
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from trafilatura import fetch_url, extract

# Download NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

class InfoBot:
    """A chatbot that retrieves information from the web."""
    
    def __init__(self):
        """Initialize the chatbot with necessary tools and conversation memory."""
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.conversation_history = []
        self.wiki_wiki = wikipediaapi.Wikipedia('InfoBot (info@example.com)', 'en')
        self.search_engines = {
            'wikipedia': self.search_wikipedia,
            'duckduckgo': self.search_duckduckgo,
            'web_scrape': self.scrape_website
        }
        self.greeting_patterns = [
            r'hello', r'hi', r'hey', r'greetings', r'howdy', r'hola'
        ]
        self.farewell_patterns = [
            r'bye', r'goodbye', r'farewell', r'see you', r'cya'
        ]
        self.question_patterns = [
            r'what is', r'who is', r'where is', r'when is', r'why is',
            r'how to', r'can you', r'could you', r'tell me about', 
            r'explain', r'information on', r'search for'
        ]
        logging.info("InfoBot initialized")
    
    def preprocess_text(self, text: str) -> List[str]:
        """Preprocess text by tokenizing, removing stop words and lemmatizing."""
        tokens = word_tokenize(text.lower())
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token.isalnum()]
        tokens = [token for token in tokens if token not in self.stop_words]
        return tokens
    
    def extract_search_query(self, message: str) -> str:
        """Extract the search query from user message."""
        # Remove question phrases to get to the core query
        cleaned_message = message.lower()
        for pattern in self.question_patterns:
            cleaned_message = re.sub(pattern, '', cleaned_message, flags=re.IGNORECASE).strip()
        
        # If cleaning removed too much, use original
        if len(cleaned_message) < 3 and len(message) > len(cleaned_message):
            return message
        return cleaned_message
    
    def categorize_message(self, message: str) -> str:
        """Categorize the user message to determine how to process it."""
        message_lower = message.lower()
        
        # Check if greeting
        for pattern in self.greeting_patterns:
            if re.search(pattern, message_lower):
                return "greeting"
        
        # Check if farewell
        for pattern in self.farewell_patterns:
            if re.search(pattern, message_lower):
                return "farewell"
        
        # Check if question/search
        for pattern in self.question_patterns:
            if re.search(pattern, message_lower):
                return "question"
        
        # Default to general conversation
        return "general"
    
    def search_wikipedia(self, query: str) -> str:
        """Search Wikipedia for the query and return a summary."""
        logging.info(f"Searching Wikipedia for: {query}")
        
        # First try exact page
        page = self.wiki_wiki.page(query)
        if page.exists():
            summary = page.summary[0:1500]  # Get first 1500 chars of summary
            if len(summary) > 0:
                return f"According to Wikipedia:\n\n{summary}\n\nLearn more: {page.fullurl}"
        
        # If exact page not found, try search
        search_results = self.wiki_wiki.opensearch(query, results=3)
        if len(search_results) > 0:
            suggestions = "\n\nRelated topics:\n" + "\n".join([f"- {title}" for title in search_results])
            first_page = self.wiki_wiki.page(search_results[0])
            if first_page.exists():
                summary = first_page.summary[0:1000]
                if len(summary) > 0:
                    return f"I found this on Wikipedia about '{search_results[0]}':\n\n{summary}\n\nLearn more: {first_page.fullurl}{suggestions}"
        
        return f"I couldn't find specific information on Wikipedia for '{query}'. Try refining your search."
    
    def search_duckduckgo(self, query: str) -> str:
        """Search DuckDuckGo for the query."""
        logging.info(f"Searching DuckDuckGo for: {query}")
        
        # DuckDuckGo doesn't have an official API, using the HTML endpoint with lite version
        url = f"https://lite.duckduckgo.com/lite/?q={query}"
        
        try:
            response = requests.get(url, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            })
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract search results
            results = []
            for a_tag in soup.find_all('a', {'class': 'result-link'}):
                results.append({
                    'title': a_tag.text.strip(),
                    'url': a_tag['href']
                })
            
            for tr in soup.find_all('tr'):
                if tr.find('a') and tr.find('td', class_='result-snippet'):
                    title_element = tr.find('a')
                    snippet_element = tr.find('td', class_='result-snippet')
                    
                    if title_element and snippet_element:
                        title = title_element.text.strip()
                        url = title_element.get('href', '')
                        snippet = snippet_element.text.strip()
                        
                        if title and url and snippet:
                            results.append({
                                'title': title,
                                'url': url,
                                'snippet': snippet
                            })
            
            # Format results
            if results:
                formatted_results = "Here's what I found on the web:\n\n"
                for i, result in enumerate(results[:5], 1):
                    title = result.get('title', 'No title')
                    url = result.get('url', 'No URL')
                    snippet = result.get('snippet', 'No description available')
                    
                    formatted_results += f"{i}. {title}\n"
                    if snippet:
                        formatted_results += f"   {snippet}\n"
                    formatted_results += f"   URL: {url}\n\n"
                
                return formatted_results
            else:
                return f"I couldn't find any search results for '{query}'. Try refining your search."
        
        except Exception as e:
            logging.error(f"Error searching DuckDuckGo: {str(e)}")
            return f"I encountered an error searching for '{query}': {str(e)}"
    
    def scrape_website(self, url: str) -> str:
        """Scrape and extract text content from a website."""
        logging.info(f"Scraping website: {url}")
        
        try:
            # Check if the URL is properly formatted
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            
            # Use trafilatura for better text extraction
            downloaded = fetch_url(url)
            if downloaded:
                text = extract(downloaded)
                if text:
                    # Limit the length of the response
                    max_length = 1500
                    if len(text) > max_length:
                        text = text[:max_length] + "...\n\n(Content truncated for readability)"
                    
                    return f"Content from {url}:\n\n{text}"
                else:
                    # Fallback to BeautifulSoup if trafilatura fails
                    return self._fallback_scrape(url)
            else:
                return f"I couldn't access the website at {url}. The site might be down or blocking requests."
        
        except Exception as e:
            logging.error(f"Error scraping website: {str(e)}")
            return f"I encountered an error when trying to read that website: {str(e)}"
    
    def _fallback_scrape(self, url: str) -> str:
        """Fallback method for scraping using BeautifulSoup if trafilatura fails."""
        try:
            response = requests.get(url, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            })
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.extract()
            
            # Get text
            text = soup.get_text()
            
            # Break into lines and remove leading and trailing space
            lines = (line.strip() for line in text.splitlines())
            # Break multi-headlines into a line each
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            # Drop blank lines
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            # Limit the length of the response
            max_length = 1500
            if len(text) > max_length:
                text = text[:max_length] + "...\n\n(Content truncated for readability)"
            
            return f"Content from {url}:\n\n{text}"
        
        except Exception as e:
            logging.error(f"Error in fallback scraping: {str(e)}")
            return f"I encountered an error when trying to read that website: {str(e)}"
    
    def generate_greeting_response(self) -> str:
        """Generate a greeting response."""
        greetings = [
            "Hello! How can I help you find information today?",
            "Hi there! I'm InfoBot. What would you like to know?",
            "Greetings! I'm here to help you find information. What are you curious about?",
            "Hey! I'm ready to search the web for you. What would you like to learn about?"
        ]
        return random.choice(greetings)
    
    def generate_farewell_response(self) -> str:
        """Generate a farewell response."""
        farewells = [
            "Goodbye! Feel free to come back if you have more questions.",
            "Farewell! I'll be here if you need me again.",
            "See you later! Come back anytime you need information.",
            "Bye for now! Don't hesitate to ask if you need to find something out."
        ]
        return random.choice(farewells)
    
    def generate_general_response(self, message: str) -> str:
        """Generate a response for general conversation."""
        general_responses = [
            "I'm a search assistant designed to find information for you. Try asking me about a topic or person.",
            "I can help you search for information on the web. What would you like to know about?",
            "I'm best at answering questions or finding information. Can you ask me something specific?",
            "I can search Wikipedia or the web for you. What topic are you interested in?"
        ]
        return random.choice(general_responses)
    
    def is_url(self, text: str) -> bool:
        """Check if the given text is a URL."""
        url_pattern = re.compile(
            r'^(?:http|https)?://'  # http:// or https:// (optional)
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain
            r'localhost|'  # localhost
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # IP
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        
        domain_pattern = re.compile(
            r'^(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)$', re.IGNORECASE)
        
        return bool(url_pattern.match(text) or domain_pattern.match(text))
    
    def get_response(self, message: str) -> str:
        """Process the user message and return a response."""
        # Add message to conversation history
        self.conversation_history.append({"user": message})
        
        # Categorize the message
        category = self.categorize_message(message)
        
        # Generate appropriate response based on category
        if category == "greeting":
            response = self.generate_greeting_response()
        elif category == "farewell":
            response = self.generate_farewell_response()
        elif category == "question":
            # First try Wikipedia
            query = self.extract_search_query(message)
            
            # Check if it's a URL first
            if self.is_url(query):
                response = self.scrape_website(query)
            else:
                # Try Wikipedia first
                response = self.search_wikipedia(query)
                
                # If Wikipedia doesn't have useful information, try DuckDuckGo
                if "couldn't find" in response:
                    response = self.search_duckduckgo(query)
        else:
            # General conversation
            response = self.generate_general_response(message)
        
        # Add response to conversation history
        self.conversation_history.append({"bot": response})
        
        # Limit conversation history to last 10 exchanges to save memory
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]
        
        return response
