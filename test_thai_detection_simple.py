#!/usr/bin/env python3
"""
Simplified test for Thai language detection without dependencies
"""
import re

def contains_thai(text: str) -> bool:
    """
    Check if text contains a significant amount of Thai characters
    
    Args:
        text: Text to check
        
    Returns:
        True if text contains significant Thai content
    """
    if not text or len(text) < 10:
        return False
        
    # Thai Unicode range (approximate)
    thai_pattern = re.compile('[\u0E00-\u0E7F]')
    
    # Count Thai characters
    thai_chars = len(thai_pattern.findall(text))
    
    # Calculate percentage of Thai characters
    if len(text) > 0:
        # Count non-whitespace characters for a more accurate percentage
        non_whitespace = len([c for c in text if not c.isspace()])
        if non_whitespace > 0:
            thai_percentage = thai_chars / non_whitespace
        else:
            thai_percentage = thai_chars / len(text)
            
        # More adaptive threshold - lower for longer text, higher for shorter text
        threshold = max(0.05, min(0.15, 10 / len(text) + 0.05))
        
        # Consider text as Thai if it contains enough Thai characters
        return thai_percentage > threshold or thai_chars > 10
    
    return False

def test_detection():
    """Test Thai language detection"""
    test_cases = [
        ("This is English text only", "English"),
        ("ทดสอบภาษาไทย", "Thai"),
        ("This has some ภาษาไทย mixed in", "Mixed"),
        ("ข้อความภาษาไทยและ English mixed", "Mixed"),
        ("บริษัท พลังงานไทย จำกัด (มหาชน)", "Thai"),
        ("A", "English (short)"),
        ("ก", "Thai (short)"),
        ("", "Empty"),
    ]
    
    print("=== Testing Thai Language Detection ===")
    for text, desc in test_cases:
        is_thai = contains_thai(text)
        print(f"Text: {text}")
        print(f"Description: {desc}")
        print(f"Contains Thai: {is_thai}")
        print()

if __name__ == "__main__":
    test_detection()
