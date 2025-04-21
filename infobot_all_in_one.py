
import re, logging, random, json, os, sys
import nltk, requests, wikipediaapi
from flask import Flask, request, jsonify, render_template_string
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams
from nltk.probability import FreqDist
from trafilatura import fetch_url, extract
from typing import List, Dict, Tuple, Optional, Union
from collections import defaultdict


logging.basicConfig(level=logging.DEBUG)
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key") 


# Download all required NLTK resources
nltk.download('punkt')  # Always download to ensure we have the latest
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')  # For better sentence parsing


class InfoBot:
    def __init__(self):
        self.lemmatizer, self.stop_words = WordNetLemmatizer(), set(stopwords.words('english'))
        self.conversation_history = []
        self.wiki_wiki = wikipediaapi.Wikipedia('InfoBot (info@example.com)', 'en')
        self.search_engines = {'wikipedia': self.search_wikipedia, 'duckduckgo': self.search_duckduckgo, 'web_scrape': self.scrape_website}
        self.greeting_patterns = [r'hello', r'hi', r'hey', r'greetings', r'howdy', r'hola']
        self.farewell_patterns = [r'bye', r'goodbye', r'farewell', r'see you', r'cya']
        self.question_patterns = [r'what is', r'who is', r'where is', r'when is', r'why is', r'how to', r'can you', 
                                r'could you', r'tell me about', r'explain', r'information on', r'search for']
        
        
        self.human_phrases = self._load_human_phrases()
        self.transition_model = self._build_transition_model()
        self.sentence_starters = self._extract_sentence_starters()
        
        logging.info("InfoBot initialized")
        
    def _load_human_phrases(self) -> List[str]:
        """Load sample human-like phrases for more natural-sounding responses."""
        return [
            "I've been looking into this, and ",
            "From what I can tell, ",
            "Let me share what I've found: ",
            "Interestingly enough, ",
            "This is actually quite fascinating - ",
            "I was just reading about this recently! ",
            "You know, it's funny you asked that because ",
            "That's a great question. ",
            "I've been thinking about this too, and ",
            "Well, according to experts, ",
            "If I understand correctly, ",
            "Let me put it this way: ",
            "The simple explanation is that ",
            "I'm glad you asked! ",
            "It turns out that ",
            "What's really interesting about this is ",
            "Many people wonder about this, and ",
            "This is something I'm passionate about! ",
            "Let me try to explain this clearly - ",
            "Here's my take on it: ",
            "Just between us, ",
            "Not many people know this, but ",
            "I should mention that ",
            "To be perfectly honest, ",
            "I was surprised to learn that ",
            "The way I see it, ",
            "This reminds me of ",
            "Actually, I just learned that ",
            "You might be surprised to hear that ",
            "Let me think about this for a second... ",
            "I'm not an expert, but I believe ",
            "From what I understand, ",
            "If you're interested in this topic, ",
            "I hope this makes sense, but ",
            "I'd like to point out that ",
            "What's particularly noteworthy is ",
            "Many people overlook the fact that ",
            "I think what you're asking about is ",
            "This is actually related to ",
            "I want to make sure I address your question properly. ",
            "So here's the deal with this - ",
            "Listen, I've got some info about this: ",
            "I'm really excited to tell you about this! ",
            "OK so check this out - ",
            "I've done some digging and found that ",
            "Here's something cool about this topic: ",
            "You're going to love learning about this - ",
            "This is honestly one of my favorite topics! ",
            "I can definitely help with that. ",
            "Let me break this down for you: "
        ]
    
    def _build_transition_model(self) -> Dict[Tuple[str, str], List[str]]:
        """Build a simple Markov chain model for text generation from predefined phrases."""
        model = defaultdict(list)
        
        human_text = " ".join([
            "I really appreciate your question about this topic.",
            "Let me share what I know about it.",
            "This is actually something I find fascinating.",
            "From what I understand, there are several important aspects to consider.",
            "Many experts in the field have different opinions on this matter.",
            "It's worth noting that this is a complex subject with various perspectives.",
            "I hope this information helps answer your question.",
            "Please let me know if you'd like me to elaborate further on any point.",
            "I'm happy to continue this conversation if you have more questions.",
            "This kind of reminds me of something I read recently about a related topic.",
            "You might also find it interesting to explore some of the connections to other areas.",
            "What I find particularly noteworthy about this is how it affects everyday life.",
            "Many people don't realize how important this actually is in the bigger picture.",
            "I think the key takeaway here is to consider multiple sources of information.",
            "Let me know if that makes sense or if you'd like me to clarify anything."
        ])
        
        words = human_text.split()
        
        for i in range(len(words) - 2):
            key = (words[i], words[i+1])
            model[key].append(words[i+2])
        
        return model
    
    def _extract_sentence_starters(self) -> List[Tuple[str, str]]:
        """Extract sentence starters from the transition model."""
        starters = []
        for (w1, w2) in self.transition_model.keys():
            if w1[0].isupper() or w1 in ["I", "i"]:
                starters.append((w1, w2))
        
        
        if not starters:
            starters = [
                ("I", "think"),
                ("Let", "me"),
                ("This", "is"),
                ("What", "I"),
                ("From", "my"),
                ("Interestingly", "enough")
            ]
        
        return starters
    
    def preprocess_text(self, text: str) -> List[str]:
        """Preprocess text by tokenizing, removing stop words and lemmatizing."""
        tokens = word_tokenize(text.lower())
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token.isalnum()]
        return [token for token in tokens if token not in self.stop_words]
    
    def extract_search_query(self, message: str) -> str:
        """Extract the search query from user message."""
        cleaned_message = message.lower()
        for pattern in self.question_patterns:
            cleaned_message = re.sub(pattern, '', cleaned_message, flags=re.IGNORECASE).strip()
        return message if len(cleaned_message) < 3 and len(message) > len(cleaned_message) else cleaned_message
    
    def categorize_message(self, message: str) -> str:
        """Categorize the user message to determine how to process it."""
        message_lower = message.lower()
        for pattern in self.greeting_patterns:
            if re.search(pattern, message_lower): return "greeting"
        for pattern in self.farewell_patterns:
            if re.search(pattern, message_lower): return "farewell"
        for pattern in self.question_patterns:
            if re.search(pattern, message_lower): return "question"
        return "general"
    
    def search_wikipedia(self, query: str) -> str:
        """Search Wikipedia for the query and return a summary."""
        logging.info(f"Searching Wikipedia for: {query}")
        
        
        page = self.wiki_wiki.page(query)
        if page.exists():
            summary = page.summary[0:1500]
            if len(summary) > 0:
                return f"{summary}\n\nSource: {page.fullurl}"
        
        
        try:
            
            search_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={query}&format=json"
            response = requests.get(search_url)
            data = response.json()
            
            search_results = []
            if 'query' in data and 'search' in data['query']:
                search_results = [result['title'] for result in data['query']['search'][:3]]
            
            if search_results:
                suggestions = "\n\nYou might also be interested in:\n" + "\n".join([f"• {title}" for title in search_results[1:]])
                first_page = self.wiki_wiki.page(search_results[0])
                if first_page.exists():
                    summary = first_page.summary[0:1000]
                    if len(summary) > 0:
                        return f"{summary}\n\nSource: {first_page.fullurl}{suggestions}"
        except Exception as e:
            logging.error(f"Error searching Wikipedia: {str(e)}")
            return f"I'm having trouble searching for information about '{query}'. Maybe try asking in a different way?"
        
        return f"I couldn't find good information about '{query}'. Could you try being more specific or asking about something else?"
    
    def search_duckduckgo(self, query: str) -> str:
        """Search DuckDuckGo for the query."""
        logging.info(f"Searching DuckDuckGo for: {query}")
        url = f"https://lite.duckduckgo.com/lite/?q={query}"
        
        try:
            response = requests.get(url, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            })
            soup = BeautifulSoup(response.text, 'html.parser')
            
            results = []
            for a_tag in soup.find_all('a', {'class': 'result-link'}):
                results.append({'title': a_tag.text.strip(), 'url': a_tag['href']})
            
            for tr in soup.find_all('tr'):
                if tr.find('a') and tr.find('td', class_='result-snippet'):
                    title_element, snippet_element = tr.find('a'), tr.find('td', class_='result-snippet')
                    if title_element and snippet_element:
                        title, url, snippet = title_element.text.strip(), title_element.get('href', ''), snippet_element.text.strip()
                        if title and url and snippet:
                            results.append({'title': title, 'url': url, 'snippet': snippet})
            
            if results:
                formatted_results = "Here's what I found on the web:\n\n"
                for i, result in enumerate(results[:5], 1):
                    title = result.get('title', 'No title')
                    url = result.get('url', 'No URL')
                    snippet = result.get('snippet', 'No description available')
                    
                    formatted_results += f"{i}. {title}\n"
                    if snippet: formatted_results += f"   {snippet}\n"
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
            if not url.startswith(('http://', 'https://')): url = 'https://' + url
            
            # More robust handling of web scraping with multiple fallbacks
            try:
                # Try trafilatura first (good for article content)
                downloaded = fetch_url(url)
                if downloaded:
                    text = extract(downloaded)
                    if text and len(text.strip()) > 100:  # Ensure we got meaningful content
                        max_length = 1500
                        if len(text) > max_length:
                            text = text[:max_length] + "...\n\n(Content truncated for readability)"
                        return f"Content from {url}:\n\n{text}"
            except Exception as trafilatura_error:
                logging.warning(f"Trafilatura scraping failed, trying fallback: {str(trafilatura_error)}")
            
            # If trafilatura fails or returns little content, try BeautifulSoup
            return self._fallback_scrape(url)
            
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
            
            for script in soup(["script", "style"]): script.extract()
            
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
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
            "Hello! How can I help you find information today? ❤️",
            "Hi there! I'm InfoBot. What would you like to know?",
            "Greetings! I'm here to help you find information. What are you curious about?",
            "Hey! I'm ready to search the web for you. What would you like to learn about?",
            "Hi there! I'm all set to help you discover new information today! 💜",
            "Hello friend! What shall we learn about today?",
            "Hey there! I'm your friendly neighborhood InfoBot! What can I help with? ❤️"
        ]
        return random.choice(greetings)
    
    def generate_farewell_response(self) -> str:
        """Generate a farewell response."""
        farewells = [
            "Goodbye! Feel free to come back if you have more questions. ❤️",
            "Farewell! I'll be here if you need me again.",
            "See you later! Come back anytime you need information. 💜",
            "Bye for now! Don't hesitate to ask if you need to find something out.",
            "Take care! I enjoyed our conversation! 💙",
            "Until next time! I'll miss our chat!",
            "Goodbye friend! Come back soon! ❤️"
        ]
        return random.choice(farewells)
    
    def generate_general_response(self, message: str) -> str:
        """Generate a response for general conversation."""
        general_responses = [
            "I'm a search assistant designed to find information for you. Try asking me about a topic or person! ❤️",
            "I can help you search for information on the web. What would you like to know about?",
            "I'm best at answering questions or finding information. Can you ask me something specific? 💜",
            "I can search Wikipedia or the web for you. What topic are you interested in?",
            "I'd love to help you learn something new today! What shall we discover together? ❤️",
            "Feel free to ask me about any topic you're curious about!",
            "I'm ready to dive into any subject you're interested in. What's on your mind? 💙"
        ]
        return random.choice(general_responses)
    
    def is_url(self, text: str) -> bool:
        """Check if the given text is a URL."""
        url_pattern = re.compile(
            r'^(?:http|https)?://'  
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  
            r'localhost|'  
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  
            r'(?::\d+)?'  
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        
        domain_pattern = re.compile(
            r'^(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)$', re.IGNORECASE)
        
        return bool(url_pattern.match(text) or domain_pattern.match(text))
    
    def generate_human_like_text(self, length=25) -> str:
        """Generate human-like text using the Markov model."""
        if not self.sentence_starters:
            return ""  # If no starters, return empty string
            
        # Choose a random starter
        current = random.choice(self.sentence_starters)
        result = [current[0], current[1]]
        
        # Generate the rest of the text
        for i in range(length):
            if (result[-2], result[-1]) in self.transition_model:
                # Get possible next words based on the last two words
                possible_next = self.transition_model[(result[-2], result[-1])]
                if possible_next:
                    # Choose a random next word
                    next_word = random.choice(possible_next)
                    result.append(next_word)
                else:
                    break
            else:
                break
        
        return " ".join(result)
        
    def humanize_response(self, response: str) -> str:
        """Make the response sound more human by adding conversational elements."""
        # Add a human-like introduction phrase
        intro = random.choice(self.human_phrases)
        
        # Split response into sentences - Use a simpler method to avoid errors
        try:
            sentences = sent_tokenize(response)
        except:
            # Fallback sentence tokenization
            sentences = [s.strip() + '.' for s in response.split('.') if s.strip()]
            if not sentences:
                sentences = [response]
        
        if not sentences:
            return response
            
        # For longer responses, also add a human-like closing phrase
        closing_phrases = [
            " I hope that helps you understand!",
            " Let me know if you have any other questions about this!",
            " Does that explain it well enough for you?",
            " What do you think about that? Pretty interesting, right?",
            " I find this topic super fascinating, don't you?",
            " I'm really curious to hear your thoughts on this too!",
            " Would you like to know more about any specific part of this?",
            " I could go on about this for hours, but I'll stop here for now!",
            " That's the simple explanation, but there's so much more to explore!",
            " I just learned about this recently myself and found it fascinating!",
            " Feel free to ask if anything isn't clear, I'm happy to explain more!"
        ]
        
        # Add emotive expressions within the text
        emotional_expressions = [
            ("\\.", " Wow!."),
            ("\\.", " Amazing!."),
            ("\\.", " Incredible!."),
            ("\\.", " Fascinating!."),
            ("\\.", " Isn't that cool?."),
            ("\\!", " So exciting!"),
            ("\\.", " That's really something!."),
            ("\\.", " Pretty mind-blowing, right?."),
            ("\\.", " I love learning about this stuff!.")
        ]
        
        # 30% chance to add an emotional expression to a random sentence
        if random.random() < 0.3 and len(sentences) > 1:
            random_idx = random.randint(0, len(sentences) - 2)  # Not the last sentence
            expression = random.choice(emotional_expressions)
            sentences[random_idx] = re.sub(expression[0] + "$", expression[1], sentences[random_idx])
        
        # Only add closing phrase for longer responses (more than 2 sentences)
        if len(sentences) > 2:
            # 85% chance to add a closing phrase (increased from 70%)
            if random.random() < 0.85:
                sentences[-1] = sentences[-1].rstrip('.') + random.choice(closing_phrases)
        
        # Occasionally add filler words or phrases between sentences
        filler_phrases = [
            " Actually, ",
            " You know what, ",
            " Interestingly enough, ",
            " By the way, ",
            " I should probably mention that ",
            " It's definitely worth noting that ",
            " Also, I think ",
            " Oh, and ",
            " This is cool - ",
            " Here's something neat: ",
            " Just between us, ",
            " Fun fact: ",
            " To be honest, ",
            " I didn't know this before, but ",
            " I've been thinking about this, and "
        ]
        
        # Add more personal thoughts/reactions to content
        personal_reactions = [
            " I personally find this really interesting because ",
            " This reminds me of ",
            " I've always wondered about this, especially ",
            " The way I see it, ",
            " From my perspective, ",
            " I think what's most fascinating here is ",
            " What stands out to me is ",
            " I'd never thought about it this way before, but "
        ]
        
        # Only add fillers for responses with multiple sentences
        if len(sentences) > 1:
            # Choose a random sentence (not the first or last) to add a filler before
            for i in range(1, min(len(sentences) - 1, 3)):  # Limit to first 3 sentences
                # 40% chance to add a filler (increased from 30%)
                if random.random() < 0.4:
                    sentences[i] = random.choice(filler_phrases) + sentences[i][0].lower() + sentences[i][1:]
            
            # 30% chance to add a personal reaction
            if random.random() < 0.3 and len(sentences) > 2:
                reaction_idx = random.randint(1, min(len(sentences) - 1, 3))
                reaction = random.choice(personal_reactions)
                if "Source:" not in sentences[reaction_idx] and "URL:" not in sentences[reaction_idx]:
                    sentences[reaction_idx] = reaction + sentences[reaction_idx][0].lower() + sentences[reaction_idx][1:]
        
        # Add extra humanizing quirks
        # 20% chance to add a typing mistake and correction
        if random.random() < 0.2:
            typos = [
                ('\\b(the)\\b', 'teh... I mean, the'),
                ('\\b(and)\\b', 'adn... sorry, and'),
                ('\\b(that)\\b', 'taht... oops, that'),
                ('\\b(with)\\b', 'wiht... *with'),
                ('\\b(information)\\b', 'informaiton... information'),
                ('\\b(because)\\b', 'becuase... because')
            ]
            typo = random.choice(typos)
            for i, sentence in enumerate(sentences):
                if re.search(typo[0], sentence) and "Source:" not in sentence and "URL:" not in sentence:
                    sentences[i] = re.sub(typo[0], typo[1], sentence, count=1)
                    break
        
        # Reassemble the response with the modifications
        # 15% chance to add a thinking pause at the beginning
        if random.random() < 0.15:
            thinking_pauses = [
                "Hmm, let me think about this... ",
                "Let me see... ",
                "Give me a second to gather my thoughts... ",
                "That's an interesting question! ",
                "Oh, I know about this! "
            ]
            intro = random.choice(thinking_pauses) + intro.lower()
        
        humanized_response = intro + sentences[0] + " " + " ".join(sentences[1:])
        
        return humanized_response
    
    def get_response(self, message: str) -> str:
        """Process the user message and return a response."""
        try:
            self.conversation_history.append({"user": message})
            category = self.categorize_message(message)
            
            if category == "greeting":
                response = self.generate_greeting_response()
            elif category == "farewell":
                response = self.generate_farewell_response()
            elif category == "question":
                query = self.extract_search_query(message)
                
                # Handle URLs separately - direct web scraping
                if self.is_url(query):
                    try:
                        raw_response = self.scrape_website(query)
                        # Don't humanize website content, just add a brief introduction
                        intros = [
                            "Here's what I found from that website: \n\n",
                            "I've extracted the following content: \n\n",
                            "Here's the information from the page: \n\n",
                            "Let me share what I found on that site: \n\n",
                            "I pulled this from the URL you shared: \n\n",
                            "I visited that site and here's what I got: \n\n"
                        ]
                        response = random.choice(intros) + raw_response
                    except Exception as e:
                        logging.error(f"Error scraping website: {str(e)}")
                        response = f"I tried to read that website, but ran into some trouble. Maybe the site is blocking automated readers or is temporarily down?"
                else:
                    # Try to get information from Wikipedia first
                    try:
                        raw_response = self.search_wikipedia(query)
                        
                        # If Wikipedia doesn't have good info, try DuckDuckGo
                        if any(phrase in raw_response for phrase in [
                            "couldn't find", "having trouble", "try being more specific"
                        ]):
                            try:
                                raw_response = self.search_duckduckgo(query)
                            except Exception as duck_error:
                                logging.error(f"Error with DuckDuckGo search: {str(duck_error)}")
                                raw_response = "I couldn't search the web right now. Maybe try a different question?"
                        
                        # Humanize the response if it contains actual information
                        if not any(phrase in raw_response for phrase in [
                                "couldn't find", "encountered an error", "try being more specific", "having trouble"
                            ]):
                            response = self.humanize_response(raw_response)
                        else:
                            # For error messages, add a simple apologetic intro
                            sorry_intros = [
                                "I'm sorry, but ",
                                "Unfortunately, ",
                                "I tried my best, but ",
                                "I wish I could help more, but ",
                                "Hmm, I'm drawing a blank here. "
                            ]
                            response = random.choice(sorry_intros) + raw_response
                    except Exception as wiki_error:
                        logging.error(f"Error with Wikipedia search: {str(wiki_error)}")
                        # Fallback to DuckDuckGo if Wikipedia errors out
                        try:
                            raw_response = self.search_duckduckgo(query)
                            if "couldn't find" not in raw_response and "error" not in raw_response:
                                response = self.humanize_response(raw_response)
                            else:
                                response = f"I couldn't find specific information about '{query}'. Could you try asking in a different way?"
                        except Exception as e:
                            logging.error(f"Error with fallback search: {str(e)}")
                            response = "I'm having trouble searching for information right now. Could we try again in a moment?"
            else:
                response = self.generate_general_response(message)
            
            self.conversation_history.append({"bot": response})
            
            if len(self.conversation_history) > 20:
                self.conversation_history = self.conversation_history[-20:]
            
            return response
            
        except Exception as e:
            logging.error(f"Unexpected error in get_response: {str(e)}")
            return "Sorry, I'm having some trouble processing your request. Let's try something else!"


infobot = InfoBot()


HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en" data-bs-theme="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>InfoBot - Web Information Chatbot</title>
    
    <!-- Bootstrap CSS - Replit Dark Theme -->
    <link rel="stylesheet" href="https://cdn.replit.com/agent/bootstrap-agent-dark-theme.min.css">
    
    <!-- Font Awesome Icons -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    
    <style>
        /* Chat container styles */
        .chat-messages {
            height: 400px;
            overflow-y: auto;
            padding: 10px;
            background-color: var(--bs-dark-bg-subtle);
            border-radius: 0.375rem;
        }
        
        /* Message styles */
        .message {
            margin-bottom: 15px;
            display: flex;
            flex-direction: column;
        }
        
        .bot-message .message-content {
            align-self: flex-start;
            background-color: var(--bs-secondary-bg);
            border-radius: 15px 15px 15px 0;
            padding: 10px 15px;
            max-width: 80%;
        }
        
        .user-message {
            align-items: flex-end;
        }
        
        .user-message .message-content {
            background-color: var(--bs-primary);
            color: white;
            border-radius: 15px 15px 0 15px;
            padding: 10px 15px;
            max-width: 80%;
        }
        
        /* Typing indicator */
        .typing-indicator {
            display: flex;
            align-items: center;
            margin-bottom: 15px;
        }
        
        .typing-indicator span {
            height: 8px;
            width: 8px;
            background-color: var(--bs-secondary);
            border-radius: 50%;
            display: inline-block;
            margin-right: 5px;
            animation: typing 1s infinite ease-in-out;
        }
        
        .typing-indicator span:nth-child(2) {
            animation-delay: 0.2s;
        }
        
        .typing-indicator span:nth-child(3) {
            animation-delay: 0.4s;
        }
        
        @keyframes typing {
            0% { transform: translateY(0px); }
            50% { transform: translateY(-5px); }
            100% { transform: translateY(0px); }
        }
        
        /* Make links in chat messages stand out */
        .message-content a {
            color: var(--bs-info);
            text-decoration: underline;
        }
        
        /* Style pre and code blocks in chat */
        .message-content pre, 
        .message-content code {
            background-color: var(--bs-tertiary-bg);
            border-radius: 4px;
            padding: 2px 4px;
            font-family: monospace;
            white-space: pre-wrap;
        }
        
        /* Better formatting for lists in chat messages */
        .message-content ul, 
        .message-content ol {
            padding-left: 20px;
        }
        
        /* Ensure the send button has proper spacing */
        #send-button {
            margin-left: 5px;
        }
        
        /* Make sure URLs don't overflow */
        .message-content {
            word-break: break-word;
        }
        
        /* Watermark styles */
        .watermark-banner {
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0px 4px 10px rgba(0,0,0,0.5);
            margin: 20px 0;
            text-align: center;
        }
        
        .watermark-text {
            font-family: 'Arial', sans-serif;
            font-weight: bold;
            color: white;
            text-shadow: 2px 2px 4px #000;
            letter-spacing: 1px;
            margin: 0;
            font-size: 24px;
        }
        
        /* Responsive adjustments */
        @media (max-width: 768px) {
            .chat-messages {
                height: 350px;
            }
            
            .bot-message .message-content,
            .user-message .message-content {
                max-width: 90%;
            }
        }
    </style>
</head>
<body>
    <div class="container py-4">
        <!-- Header -->
        <header class="pb-3 mb-4 border-bottom">
            <div class="d-flex align-items-center">
                <i class="fas fa-robot text-info me-3 fs-1"></i>
                <h1 class="fs-4">InfoBot</h1>
            </div>
            <p class="text-muted">Your Web-Powered Information Assistant</p>
        </header>
        
        <!-- Large Watermark Banner -->
        <div class="watermark-banner mb-4">
            <h2 class="watermark-text">Kartikeya Lakhera (will be removed once work is done)</h2>
        </div>
        
        <!-- Chat Container -->
        <div class="card mb-4">
            <div class="card-header">
                <i class="fas fa-comments me-2"></i>
                Conversation
            </div>
            <div class="card-body">
                <div id="chat-messages" class="chat-messages mb-3">
                    <!-- Welcome message -->
                    <div class="message bot-message">
                        <div class="message-content">
                            <p>Hello! I'm InfoBot, your information assistant. I can help you find information from the web.</p>
                            <p>Here are some things you can ask me:</p>
                            <ul>
                                <li>What is quantum computing?</li>
                                <li>Who was Marie Curie?</li>
                                <li>Tell me about the history of jazz</li>
                                <li>How does solar power work?</li>
                            </ul>
                            <p>You can also ask me to read a webpage for you by providing the URL.</p>
                        </div>
                    </div>
                </div>
                
                <!-- Input Area -->
                <div class="input-group">
                    <input type="text" id="user-input" class="form-control" placeholder="Ask me anything...">
                    <button id="send-button" class="btn btn-primary">
                        <i class="fas fa-paper-plane"></i> Send
                    </button>
                </div>
            </div>
        </div>
        
        <!-- Info Cards -->
        <div class="row g-4">
            <div class="col-md-4">
                <div class="card h-100">
                    <div class="card-body">
                        <h5 class="card-title"><i class="fas fa-search me-2 text-primary"></i>Web Search</h5>
                        <p class="card-text">InfoBot can search the web for information and provide you with relevant results.</p>
                    </div>
                </div>
            </div>
            <div class="col-md-4">
                <div class="card h-100">
                    <div class="card-body">
                        <h5 class="card-title"><i class="fas fa-book me-2 text-success"></i>Wikipedia Access</h5>
                        <p class="card-text">Ask about any topic to get information directly from Wikipedia's vast knowledge base.</p>
                    </div>
                </div>
            </div>
            <div class="col-md-4">
                <div class="card h-100">
                    <div class="card-body">
                        <h5 class="card-title"><i class="fas fa-globe me-2 text-info"></i>Website Reader</h5>
                        <p class="card-text">Provide a URL and InfoBot will read and summarize the content for you.</p>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Footer -->
        <footer class="pt-3 mt-4 text-muted border-top">
            <div class="d-flex justify-content-between align-items-center">
                <div>
                    <small>InfoBot &copy; 2023 <span style="color: #ff6b6b;">❤</span></small>
                </div>
                <div>
                    <span class="badge bg-secondary">All-In-One Python Web Chatbot</span>
                </div>
            </div>
            <div class="text-center mt-2">
                <div class="watermark" style="font-family: 'Arial', sans-serif; opacity: 1; font-style: italic; font-size: 18px; font-weight: bold; padding: 10px; background-color: rgba(50, 50, 50, 0.7); border-radius: 5px; margin-top: 5px;">
                    <span style="color: #77ddff; text-shadow: 1px 1px 2px #000;">Kartikeya Lakhera (will be removed once work is done)</span>
                </div>
            </div>
        </footer>
    </div>
    
    <!-- Bootstrap JS -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    
    <script>
        document.addEventListener('DOMContentLoaded', function() {
            const chatMessages = document.getElementById('chat-messages');
            const userInput = document.getElementById('user-input');
            const sendButton = document.getElementById('send-button');
            
            function addMessage(text, isUser = false) {
                const messageDiv = document.createElement('div');
                messageDiv.className = isUser ? 'message user-message' : 'message bot-message';
                
                const contentDiv = document.createElement('div');
                contentDiv.className = 'message-content';
                
                const formattedText = formatMessageText(text);
                contentDiv.innerHTML = formattedText;
                
                messageDiv.appendChild(contentDiv);
                chatMessages.appendChild(messageDiv);
                
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }
            
            function formatMessageText(text) {
                // Use a safer regex construction to avoid errors
                const urlRegex = new RegExp("(https?:\\/\\/[^\\s]+)", "g");
                text = text.replace(urlRegex, url => `<a href="${url}" target="_blank">${url}</a>`);
                return text.replace(new RegExp("\\n", "g"), '<br>');
            }
            
            function showTypingIndicator() {
                const typingDiv = document.createElement('div');
                typingDiv.className = 'typing-indicator';
                typingDiv.innerHTML = `
                    <div class="message-content">
                        <span></span>
                        <span></span>
                        <span></span>
                    </div>
                `;
                typingDiv.id = 'typing-indicator';
                chatMessages.appendChild(typingDiv);
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }
            
            function removeTypingIndicator() {
                const typingIndicator = document.getElementById('typing-indicator');
                if (typingIndicator) {
                    typingIndicator.remove();
                }
            }
            
            async function sendMessage(message) {
                try {
                    showTypingIndicator();
                    
                    const response = await fetch('/chat', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ message: message }),
                    });
                    
                    const data = await response.json();
                    removeTypingIndicator();
                    
                    if (response.ok) {
                        addMessage(data.response);
                    } else {
                        addMessage("I'm sorry, I couldn't process your request. Please try again.");
                    }
                } catch (error) {
                    removeTypingIndicator();
                    addMessage("I'm sorry, there was an error connecting to the server. Please check your connection and try again.");
                    console.error('Error:', error);
                }
            }
            
            sendButton.addEventListener('click', function() {
                const message = userInput.value.trim();
                if (message) {
                    addMessage(message, true);
                    userInput.value = '';
                    sendMessage(message);
                }
            });
            
            userInput.addEventListener('keypress', function(e) {
                if (e.key === 'Enter') {
                    const message = userInput.value.trim();
                    if (message) {
                        addMessage(message, true);
                        userInput.value = '';
                        sendMessage(message);
                    }
                }
            });
            
            userInput.focus();
        });
    </script>
</body>
</html>
'''


@app.route('/')
def index():
    """Render the main chat interface."""
    return render_template_string(HTML_TEMPLATE)

@app.route('/chat', methods=['POST'])
def chat():
    """Process chat requests from the user."""
    user_message = request.json.get('message', '')
    
    if not user_message:
        return jsonify({'response': 'Please provide a message.'})
    
    try:
        
        response = infobot.get_response(user_message)
        return jsonify({'response': response})
    except Exception as e:
        logging.error(f"Error processing request: {str(e)}")
        return jsonify({'response': f"I'm sorry, I encountered an error: {str(e)}"})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)