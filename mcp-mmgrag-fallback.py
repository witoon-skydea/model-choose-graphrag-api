#!/usr/bin/env python3
"""
Model-Choose GraphRAG - MCP Server Implementation (Fallback Version)

This version of the MCP server uses in-memory storage only,
avoiding filesystem access issues.
"""
import os
import sys
import json
import uuid
import tempfile
from typing import List, Dict, Any, Optional
from mcp.server.fastmcp import FastMCP

# Create MCP server instance
mcp = FastMCP("Model-Choose GraphRAG MCP Server (Fallback)")

# In-memory storage for companies
COMPANIES = {
    "default": {
        "id": "default",
        "name": "Default Company",
        "description": "Default RAG system company",
        "active": True
    }
}

ACTIVE_COMPANY = "default"
DEFAULT_LLM_MODEL = "llama3:8b"
DEFAULT_EMBEDDING_MODEL = "mxbai-embed-large:latest"

# Define MCP tools
@mcp.tool()
def list_companies() -> str:
    """แสดงรายชื่อบริษัททั้งหมด"""
    company_list = []
    for company_id, company in COMPANIES.items():
        company_list.append({
            "id": company_id,
            "name": company["name"],
            "description": company["description"],
            "active": company_id == ACTIVE_COMPANY
        })
    return json.dumps(company_list, ensure_ascii=False, indent=2)

@mcp.tool()
def add_company(name: str) -> str:
    """เพิ่มบริษัทใหม่"""
    global COMPANIES
    company_id = name.lower().replace(" ", "_")
    
    if company_id in COMPANIES:
        return json.dumps({"success": False, "error": f"บริษัท {company_id} มีอยู่แล้ว"}, ensure_ascii=False, indent=2)
    
    COMPANIES[company_id] = {
        "id": company_id,
        "name": name,
        "description": "",
        "active": False
    }
    
    return json.dumps({"success": True, "id": company_id, "name": name}, ensure_ascii=False, indent=2)

@mcp.tool()
def select_company(name: str) -> str:
    """เลือกบริษัทที่ต้องการทำงานด้วย"""
    global ACTIVE_COMPANY
    
    if name not in COMPANIES:
        return json.dumps({"success": False, "error": f"ไม่พบบริษัท {name}"}, ensure_ascii=False, indent=2)
    
    ACTIVE_COMPANY = name
    
    return json.dumps({"success": True, "message": f"บริษัท '{name}' ถูกตั้งค่าเป็นบริษัทที่ใช้งานอยู่"}, ensure_ascii=False, indent=2)

@mcp.tool()
def get_current_company() -> str:
    """แสดงบริษัทที่กำลังเลือกอยู่"""
    company = COMPANIES.get(ACTIVE_COMPANY, {})
    company["active"] = True
    return json.dumps(company, ensure_ascii=False, indent=2)

@mcp.tool()
def add_file(company_name: str, file_path: str, description: str = "") -> str:
    """เพิ่มไฟล์ให้กับบริษัท (ระบบจะจำลองการเพิ่มไฟล์)"""
    if company_name not in COMPANIES:
        return json.dumps({"success": False, "error": f"ไม่พบบริษัท {company_name}"}, ensure_ascii=False, indent=2)
    
    if not os.path.exists(file_path):
        return json.dumps({"success": False, "error": f"ไม่พบไฟล์ {file_path}"}, ensure_ascii=False, indent=2)
    
    # Just simulate file processing, don't actually process
    return json.dumps({
        "success": True, 
        "message": f"จำลองการเพิ่มไฟล์ {os.path.basename(file_path)} ให้กับบริษัท {company_name}",
        "company_id": company_name,
        "file": os.path.basename(file_path)
    }, ensure_ascii=False, indent=2)

@mcp.tool()
def add_chat_text(company_name: str, text: str, description: str = "") -> str:
    """เพิ่มข้อความแชทให้กับบริษัท (ระบบจะจำลองการเพิ่มข้อความ)"""
    if company_name not in COMPANIES:
        return json.dumps({"success": False, "error": f"ไม่พบบริษัท {company_name}"}, ensure_ascii=False, indent=2)
    
    # Truncate text for display if it's too long
    display_text = text[:50] + "..." if len(text) > 50 else text
    
    # Just simulate text processing
    return json.dumps({
        "success": True, 
        "message": f"จำลองการเพิ่มข้อความ '{display_text}' ให้กับบริษัท {company_name}",
        "company_id": company_name,
        "text_length": len(text)
    }, ensure_ascii=False, indent=2)

@mcp.tool()
def retrieve_chunks(company_name: str, limit: int = 10, data_type: str = "all") -> str:
    """ดึงข้อมูลชังค์จากบริษัท (ระบบจะจำลองการดึงข้อมูล)"""
    if company_name not in COMPANIES:
        return json.dumps({"success": False, "error": f"ไม่พบบริษัท {company_name}"}, ensure_ascii=False, indent=2)
    
    # Simulate chunks retrieval
    chunks = [
        f"[example] นี่คือตัวอย่างชังค์ที่ 1 สำหรับบริษัท {company_name} (ประเภท: {data_type})",
        f"[example] นี่คือตัวอย่างชังค์ที่ 2 สำหรับบริษัท {company_name} (ประเภท: {data_type})",
        f"[example] นี่คือตัวอย่างชังค์ที่ 3 สำหรับบริษัท {company_name} (ประเภท: {data_type})"
    ]
    
    return "\n\n---\n\n".join(chunks[:limit])

@mcp.tool()
def retrieve_raw_chunks(company_name: str, limit: int = 10, data_type: str = "all") -> str:
    """ดึงข้อมูลชังค์ดิบจากบริษัท (ระบบจะจำลองการดึงข้อมูล)"""
    if company_name not in COMPANIES:
        return json.dumps({"success": False, "error": f"ไม่พบบริษัท {company_name}"}, ensure_ascii=False, indent=2)
    
    # Simulate raw chunks
    chunks = []
    for i in range(min(limit, 3)):
        chunks.append({
            "type": data_type if data_type != "all" else ["vector", "graph"][i % 2],
            "index": i + 1,
            "content": f"นี่คือตัวอย่างชังค์ที่ {i+1} สำหรับบริษัท {company_name}",
            "source": "example_source"
        })
    
    return json.dumps(chunks, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    print("Running fallback version of GraphRAG MCP Server (in-memory only)", file=sys.stderr)
    mcp.run(transport="stdio")
