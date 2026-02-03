import xml.etree.ElementTree as ET
import sys
import os

def extract_text(xml_path):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Namespaces are annoying in DOCX, usually 'w' is http://schemas.openxmlformats.org/wordprocessingml/2006/main
        # But we can just iterate all elements and look for 't' (text) tags, or 'p' (paragraph) tags
        
        # A simple recursive text extraction might be best
        text_content = []
        for elem in root.iter():
            if elem.tag.endswith('}t'):
                if elem.text:
                    text_content.append(elem.text)
            elif elem.tag.endswith('}p'):
                text_content.append('\n') # Newline for paragraphs
        
        print("".join(text_content))
        
    except Exception as e:
        print(f"Error parsing XML: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_docx_text.py <path_to_document.xml>")
    else:
        extract_text(sys.argv[1])
