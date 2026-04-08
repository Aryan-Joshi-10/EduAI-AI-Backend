"""
Utility functions used across services.
"""
import re
import json
import logging

logger = logging.getLogger(__name__)


def clean_json_string(text: str) -> str:
    """Clean JSON string by fixing common escape sequence issues - robust version"""
    if not text:
        return text
    
    # Remove markdown code blocks if present
    text = re.sub(r'```json\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'```\s*', '', text)
    
    # Process character by character to fix invalid escapes
    # Valid JSON escapes: \" \\ \/ \b \f \n \r \t \uXXXX
    result = []
    i = 0
    
    while i < len(text):
        if text[i] == '\\' and i + 1 < len(text):
            next_char = text[i + 1]
            # Check if it's a valid escape sequence
            if next_char in '"\\/bfnrt':
                # Valid single-character escape - keep as is
                result.append('\\' + next_char)
                i += 2
            elif next_char == 'u':
                # Check for valid unicode escape \uXXXX
                if i + 5 < len(text):
                    hex_part = text[i + 2:i + 6]
                    if len(hex_part) == 4 and all(c in '0123456789abcdefABCDEF' for c in hex_part):
                        # Valid unicode escape - keep as is
                        result.append(text[i:i + 6])
                        i += 6
                    else:
                        # Invalid unicode escape - output \\u
                        result.append('\\\\')
                        result.append('u')
                        i += 2
                else:
                    # Incomplete unicode escape - output \\u
                    result.append('\\\\')
                    result.append('u')
                    i += 2
            else:
                # Invalid escape sequence - output \\ + char
                result.append('\\\\')
                result.append(next_char)
                i += 2
        else:
            result.append(text[i])
            i += 1
    
    # Handle trailing backslash
    if i < len(text) and text[i] == '\\':
        result.append('\\\\')
    
    text = ''.join(result)
    
    # Fix trailing commas before closing braces/brackets
    text = re.sub(r',(\s*[}\]])', r'\1', text)
    
    # Fix unterminated strings by closing them at the end of JSON structure
    open_quotes = text.count('"') - text.count('\\"')
    if open_quotes % 2 != 0:
        # Odd number of quotes - likely unterminated string
        if not text.rstrip().endswith('"') and text.rstrip().endswith('}'):
            last_quote_pos = text.rfind('"')
            if last_quote_pos > 0:
                after_quote = text[last_quote_pos + 1:].strip()
                if after_quote and not after_quote.startswith(','):
                    text = re.sub(r'([^"])\s*\}', r'\1"\}', text, count=1)
    
    return text


def clean_latex_content(text):
    """
    Cleans text content for proper LaTeX rendering on the frontend.
    1. Removes newlines immediately after equals signs.
    2. Replaces non-standard \degree with ^{\circ}.
    3. Ensures strictly standard spacing.
    4. Merges MCQ option labels with their content (e.g., "a)\nValue" -> "a) Value").
    """
    if not isinstance(text, str):
        return text

    text = re.sub(r'=\s*[\r\n]+\s*', '= ', text)
    # Replace \degree with ^{\circ}
    # Use str.replace() for simple literal string replacement to avoid regex escape issues
    # This is more reliable than regex replacement for literal strings
    text = text.replace('\\degree', '^{\\circ}')
    text = re.sub(r'([a-dA-D]\))\s*[\r\n]+\s*', r'\1 ', text)
    return text.strip()


def parse_xml_to_json(xml_text: str) -> dict:
    """
    Parse XML-like tagged blocks to JSON format.
    Handles both single page and multi-page formats.
    """
    if not xml_text:
        return {}
    
    try:
        # Remove any markdown code blocks if present
        xml_text = re.sub(r'```xml\s*', '', xml_text, flags=re.IGNORECASE)
        xml_text = re.sub(r'```\s*', '', xml_text)
        xml_text = xml_text.strip()
        
        # Try to parse as multi-page format first
        pages_pattern = r'<pages>(.*?)</pages>'
        pages_match = re.search(pages_pattern, xml_text, re.DOTALL | re.IGNORECASE)
        
        if pages_match:
            pages_content = pages_match.group(1)
            # Extract all page blocks
            page_pattern = r'<page\s+number=["\']?(\d+)["\']?\s*>(.*?)</page>'
            page_matches = re.finditer(page_pattern, pages_content, re.DOTALL | re.IGNORECASE)
            
            pages_list = []
            for page_match in page_matches:
                page_num = int(page_match.group(1))
                page_content = page_match.group(2)
                
                # Extract elements from this page
                elements = parse_elements_from_xml(page_content)
                if elements:
                    pages_list.append({
                        "page_number": page_num,
                        "elements": elements
                    })
            
            if pages_list:
                return {"pages": pages_list}
        
        # Try single page format (elements)
        elements_pattern = r'<elements>(.*?)</elements>'
        elements_match = re.search(elements_pattern, xml_text, re.DOTALL | re.IGNORECASE)
        
        if elements_match:
            elements_content = elements_match.group(1)
            elements = parse_elements_from_xml(elements_content)
            if elements:
                return {"elements": elements}
        
        # If no structured format found, try to extract any element tags directly
        elements = parse_elements_from_xml(xml_text)
        if elements:
            return {"elements": elements}
        
        return {}
    
    except Exception as e:
        logger.debug(f"Error parsing XML to JSON: {e}")
        return {}


def parse_elements_from_xml(xml_content: str) -> list:
    """
    Extract element tags from XML content.
    Handles CDATA sections and escaped content.
    """
    elements = []
    
    # Pattern to match element tags with optional CDATA
    element_pattern = r'<element\s+type=["\']([^"\']+)["\']\s*>(.*?)</element>'
    element_matches = re.finditer(element_pattern, xml_content, re.DOTALL | re.IGNORECASE)
    
    for match in element_matches:
        try:
            el_type = match.group(1).strip()
            el_content = match.group(2).strip()
            
            # Handle CDATA sections
            if el_content.startswith('<![CDATA[') and el_content.endswith(']]>'):
                el_content = el_content[9:-3]  # Remove CDATA wrapper
            
            # Clean up content (remove extra whitespace, decode entities if needed)
            el_content = re.sub(r'\s+', ' ', el_content).strip()
            
            # Basic HTML/XML entity decoding
            el_content = el_content.replace('&lt;', '<')
            el_content = el_content.replace('&gt;', '>')
            el_content = el_content.replace('&amp;', '&')
            el_content = el_content.replace('&quot;', '"')
            el_content = el_content.replace('&apos;', "'")
            
            if el_type and el_content:
                elements.append({
                    "type": el_type,
                    "content": el_content
                })
        except Exception as e:
            logger.debug(f"Error parsing element: {e}")
            continue
    
    return elements

