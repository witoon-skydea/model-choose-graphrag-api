"""
Thai-specific entity extraction module for GraphRAG
"""
import json
from typing import List, Dict, Any, Tuple
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

THAI_ENTITY_EXTRACTION_PROMPT = """
คุณเป็นผู้เชี่ยวชาญในการสกัดเอนทิตีและความสัมพันธ์จากเอกสารภาษาไทย
กรุณาวิเคราะห์ข้อความต่อไปนี้และสกัด:
1. เอนทิตีที่สำคัญทั้งหมด (คน, องค์กร, สถานที่, แนวคิด ฯลฯ)
2. ความสัมพันธ์ระหว่างเอนทิตีเหล่านี้

โปรดตอบกลับในรูปแบบ JSON เท่านั้น ไม่ต้องมีคำอธิบายเพิ่มเติม:
{{
  "entities": [
    {{"id": "person_1", "name": "ชื่อบุคคล", "type": "person"}},
    {{"id": "org_1", "name": "ชื่อองค์กร", "type": "organization"}}
  ],
  "relations": [
    {{"source": "person_1", "target": "org_1", "type": "works_for"}}
  ]
}}

ข้อควรระวัง:
- สร้าง ID ที่เป็นเอกลักษณ์และสั้นแต่เข้าใจง่าย (เช่น "pm_prayut", "mot", "bangkok")
- สกัดเฉพาะความสัมพันธ์ที่มีการระบุไว้อย่างชัดเจน ไม่คาดเดา
- ระบุประเภทความสัมพันธ์ให้เฉพาะเจาะจง (เช่น "ทำงานให้", "ตั้งอยู่ใน", "เป็นผู้กำกับดูแล")
- หากเป็นเอกสารนโยบาย ระบุความสัมพันธ์ระหว่างหน่วยงานและนโยบาย

เนื้อหาที่ต้องวิเคราะห์:
{text}

กรุณาตอบกลับเป็น JSON:
"""

def extract_thai_entities_relations(llm, text: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Extract entities and relations from Thai text using LLM
    
    Args:
        llm: LLM model
        text: Text to analyze
        
    Returns:
        Tuple of (entities, relations)
    """
    # Check if text is empty or too short
    if not text or len(text.strip()) < 10:
        print("Text is too short for Thai entity extraction")
        return [], []
    
    # Create a direct prompt for JSON generation
    prompt = PromptTemplate(
        template=THAI_ENTITY_EXTRACTION_PROMPT,
        input_variables=["text"]
    )
    
    # Create chain
    chain = LLMChain(llm=llm, prompt=prompt)
    
    # Run chain with error handling
    try:
        response = chain.invoke({"text": text})
        result_text = response["text"]
    except Exception as e:
        print(f"Error in Thai entity extraction: {e}")
        # Use a simpler approach as fallback
        try:
            # Direct question to the model
            simple_prompt = f"""Extract main entities (people, organizations, places) from this Thai text and format as JSON:
            {{
              "entities": [
                {{"id": "1", "name": "Example Name", "type": "person"}}
              ]
            }}
            
            Text: {text[:1000]}
            
            JSON:"""
            
            fallback_response = llm.invoke(simple_prompt)
            result_text = fallback_response
        except Exception as fallback_error:
            print(f"Fallback extraction also failed: {fallback_error}")
            return [], []
    
# Extract JSON from the response
    try:
        # Try to find JSON between curly braces
        if "{" in result_text and "}" in result_text:
            start_idx = result_text.find("{")
            end_idx = result_text.rfind("}") + 1
            json_str = result_text[start_idx:end_idx]
            
            try:
                result = json.loads(json_str)
            except json.JSONDecodeError:
                # Try cleaning the JSON string
                import re
                json_str = re.sub(r'```json\s*|\s*```', '', json_str)  # Remove markdown code blocks
                json_str = re.sub(r',\s*}', '}', json_str)  # Remove trailing commas
                json_str = re.sub(r',\s*]', ']', json_str)  # Remove trailing commas in arrays
                try:
                    result = json.loads(json_str)
                except:
                    print("Could not parse JSON even after cleanup")
                    return [], []
            
            # Extract entities and relations
            entities = result.get("entities", [])
            relations = result.get("relations", [])
            
            # Add default IDs if missing
            for i, entity in enumerate(entities):
                if "id" not in entity and "name" in entity:
                    # Create a simple ID from the name
                    name = entity["name"].lower()
                    # Replace Thai characters with romanized versions where possible
                    name = ''.join(c if c.isalnum() else '_' for c in name)
                    entity["id"] = f"{entity.get('type', 'entity')}_{name}_{i}"
                
                # Ensure attributes exist
                if "attributes" not in entity:
                    entity["attributes"] = {}
                
                # Add original Thai name as attribute
                entity["attributes"]["thai_name"] = entity["name"]
            
            # Verify source and target exist
            valid_relations = []
            entity_ids = {entity.get("id") for entity in entities}
            
            for relation in relations:
                source = relation.get("source")
                target = relation.get("target")
                
                if source in entity_ids and target in entity_ids:
                    # Ensure attributes exist
                    if "attributes" not in relation:
                        relation["attributes"] = {}
                    valid_relations.append(relation)
            
            return entities, valid_relations
        else:
            print("No JSON object found in the response")
            return [], []
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        print(f"Error parsing Thai entity extraction response: {e}")
        print(f"Response: {result_text[:100]}...")  # Print just the first 100 chars
        return [], []

def extract_thai_query_entities(llm, query: str) -> List[str]:
    """
    Extract entities from a Thai query for improved knowledge graph retrieval
    
    Args:
        llm: LLM model
        query: Query to extract entities from
        
    Returns:
        List of entity names
    """
    # Check if query is empty or too short
    if not query or len(query.strip()) < 5:
        return []
    
    # Create a prompt for extraction
    prompt = f"""
    กรุณาสกัดคำสำคัญที่เป็นชื่อเฉพาะ หัวข้อ หรือคำหลักจากคำถามนี้
    ส่งกลับเป็น JSON array เช่น ["คำ1", "คำ2"] เท่านั้น ไม่ต้องมีคำอธิบายเพิ่มเติม

    คำถาม: {query}

    คำสำคัญ (เฉพาะ JSON array):
    """
    
    try:
        response = llm.invoke(prompt)
        
        # Try to find JSON array in response
        if "[" in response and "]" in response:
            json_str = response[response.find("["):response.rfind("]")+1]
            try:
                entities = json.loads(json_str)
                if isinstance(entities, list):
                    return [str(e) for e in entities if e]  # Convert all to strings and filter empty
                else:
                    return [str(entities)]  # Handle case where a single entity is returned
            except json.JSONDecodeError:
                # Fall back to simple extraction
                stripped = json_str.strip("[]")
                if "," in stripped:
                    return [e.strip().strip('"\'') for e in stripped.split(",") if e.strip()]
                else:
                    return [stripped.strip().strip('"\'')]
        else:
            # No JSON array found, use keywords from query
            import re
            # Find Thai words (consecutive Thai characters)
            thai_pattern = re.compile(r'[\u0E00-\u0E7F]+')
            thai_words = thai_pattern.findall(query)
            
            # If Thai words found, use them
            if thai_words:
                return thai_words[:3]  # Return first 3 Thai words
            else:
                # Otherwise return longest words from query
                words = sorted([w for w in query.split() if len(w) > 1], key=len, reverse=True)
                return words[:3]  # Return 3 longest words
    except Exception as e:
        print(f"Error extracting Thai query entities: {e}")
        # Fallback: return longest words from the query
        words = sorted([w for w in query.split() if len(w) > 1], key=len, reverse=True)
        return words[:3]  # Return 3 longest words
