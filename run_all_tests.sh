#!/bin/bash
# ทดสอบระบบ GraphRAG API อัตโนมัติ

# สีสำหรับการแสดงผล
RED="\033[0;31m"
GREEN="\033[0;32m"
YELLOW="\033[0;33m"
BLUE="\033[0;34m"
RESET="\033[0m"

echo -e "${BLUE}=== ทดสอบระบบ GraphRAG API สำหรับภาษาไทย ===${RESET}"
echo

# ตรวจสอบว่ามีไฟล์ที่ต้องการหรือไม่
required_files=(
    "test_thai_support.py"
    "demo_thai_support.py"
    "check_prerequisites.py"
)

for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo -e "${RED}ไม่พบไฟล์: $file${RESET}"
        echo -e "${YELLOW}กรุณาตรวจสอบว่าได้ติดตั้งไฟล์ทั้งหมดแล้ว${RESET}"
        exit 1
    fi
done

# ทำให้สคริปต์สามารถทำงานได้
chmod +x test_thai_support.py demo_thai_support.py check_prerequisites.py

# ตรวจสอบความพร้อมของระบบ
echo -e "${BLUE}ตรวจสอบความพร้อมของระบบ...${RESET}"
python check_prerequisites.py

if [ $? -ne 0 ]; then
    echo -e "${YELLOW}พบปัญหาในการตรวจสอบความพร้อม กรุณาติดตั้งโมดูลที่จำเป็นก่อน${RESET}"
    echo -e "${YELLOW}คุณสามารถรัน ./install_thai_support.sh เพื่อติดตั้งโมดูลที่จำเป็น${RESET}"
    
    # ถามผู้ใช้ว่าต้องการดำเนินการต่อหรือไม่
    read -p "ต้องการดำเนินการต่อหรือไม่? (y/n): " continue_test
    if [[ $continue_test != "y" && $continue_test != "Y" ]]; then
        echo -e "${RED}ยกเลิกการทดสอบ${RESET}"
        exit 1
    fi
fi

# สร้างโฟลเดอร์สำหรับผลการทดสอบ
TEST_OUTPUT_DIR="test_results"
mkdir -p $TEST_OUTPUT_DIR

# 1. ทดสอบการตรวจจับภาษาไทย
echo -e "\n${BLUE}ทดสอบการตรวจจับภาษาไทย...${RESET}"
python -c "from rag.knowledge_graph.graph import contains_thai; print(f'ภาษาไทย: {contains_thai(\"นี่คือข้อความภาษาไทย\")}'); print(f'ภาษาอังกฤษ: {contains_thai(\"This is English text\")}'); print(f'ผสม: {contains_thai(\"This has ภาษาไทย mixed in\")}')" > "$TEST_OUTPUT_DIR/thai_detection_test.log" 2>&1

if [ $? -eq 0 ]; then
    echo -e "${GREEN}การทดสอบการตรวจจับภาษาไทยสำเร็จ${RESET}"
    grep -q "ภาษาไทย: True" "$TEST_OUTPUT_DIR/thai_detection_test.log" && echo -e "${GREEN}- ตรวจจับภาษาไทยถูกต้อง${RESET}" || echo -e "${RED}- ตรวจจับภาษาไทยไม่ถูกต้อง${RESET}"
    grep -q "ภาษาอังกฤษ: False" "$TEST_OUTPUT_DIR/thai_detection_test.log" && echo -e "${GREEN}- ตรวจจับภาษาอังกฤษถูกต้อง${RESET}" || echo -e "${RED}- ตรวจจับภาษาอังกฤษไม่ถูกต้อง${RESET}"
    grep -q "ผสม: True" "$TEST_OUTPUT_DIR/thai_detection_test.log" && echo -e "${GREEN}- ตรวจจับภาษาผสมถูกต้อง${RESET}" || echo -e "${RED}- ตรวจจับภาษาผสมไม่ถูกต้อง${RESET}"
else
    echo -e "${RED}การทดสอบการตรวจจับภาษาไทยล้มเหลว${RESET}"
fi

# 2. ทดสอบชุดทดสอบหลัก
echo -e "\n${BLUE}กำลังรันชุดทดสอบภาษาไทย...${RESET}"
python test_thai_support.py > "$TEST_OUTPUT_DIR/thai_support_test.log" 2>&1

if [ $? -eq 0 ]; then
    echo -e "${GREEN}การทดสอบชุดทดสอบภาษาไทยสำเร็จ${RESET}"
    # ตรวจสอบผลการทดสอบเพิ่มเติม
    grep -q "Contains Thai: True" "$TEST_OUTPUT_DIR/thai_support_test.log" && echo -e "${GREEN}- ตรวจจับภาษาไทยในการทดสอบสำเร็จ${RESET}" || echo -e "${YELLOW}- ไม่พบผลการตรวจจับภาษาไทยในการทดสอบ${RESET}"
    grep -q "Extracted entities" "$TEST_OUTPUT_DIR/thai_support_test.log" && echo -e "${GREEN}- สกัดเอนทิตีในการทดสอบสำเร็จ${RESET}" || echo -e "${YELLOW}- ไม่พบผลการสกัดเอนทิตีในการทดสอบ${RESET}"
else
    echo -e "${RED}การทดสอบชุดทดสอบภาษาไทยล้มเหลว${RESET}"
    echo -e "${YELLOW}ดูรายละเอียดเพิ่มเติมได้ที่: $TEST_OUTPUT_DIR/thai_support_test.log${RESET}"
fi

# 3. รันการสาธิตถ้ามี Ollama
echo -e "\n${BLUE}ตรวจสอบการเชื่อมต่อกับ Ollama...${RESET}"
if curl -s -o /dev/null -w "%{http_code}" http://localhost:11434/api/version | grep -q "200"; then
    echo -e "${GREEN}สามารถเชื่อมต่อกับ Ollama ได้${RESET}"
    echo -e "${BLUE}กำลังรันการสาธิตภาษาไทย...${RESET}"
    python demo_thai_support.py > "$TEST_OUTPUT_DIR/thai_demo.log" 2>&1
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}การสาธิตภาษาไทยสำเร็จ${RESET}"
        # ตรวจสอบผลการสาธิตเพิ่มเติม
        grep -q "Knowledge graph" "$TEST_OUTPUT_DIR/thai_demo.log" && echo -e "${GREEN}- สร้างกราฟความรู้สำเร็จ${RESET}" || echo -e "${YELLOW}- ไม่พบผลการสร้างกราฟความรู้${RESET}"
        grep -q "Generated response" "$TEST_OUTPUT_DIR/thai_demo.log" && echo -e "${GREEN}- สร้างคำตอบสำเร็จ${RESET}" || echo -e "${YELLOW}- ไม่พบผลการสร้างคำตอบ${RESET}"
    else
        echo -e "${RED}การสาธิตภาษาไทยล้มเหลว${RESET}"
        echo -e "${YELLOW}ดูรายละเอียดเพิ่มเติมได้ที่: $TEST_OUTPUT_DIR/thai_demo.log${RESET}"
    fi
else
    echo -e "${YELLOW}ไม่สามารถเชื่อมต่อกับ Ollama ได้ ข้ามการรันการสาธิต${RESET}"
    echo -e "${YELLOW}คุณสามารถรันการสาธิตด้วยตนเองได้ด้วยคำสั่ง: python demo_thai_support.py${RESET}"
fi

echo -e "\n${BLUE}สรุปผลการทดสอบ${RESET}"
echo -e "${BLUE}====================${RESET}"
echo -e "ผลลัพธ์การทดสอบทั้งหมดอยู่ที่: ${GREEN}$TEST_OUTPUT_DIR${RESET}"
echo -e "โปรดตรวจสอบไฟล์บันทึกเพื่อดูรายละเอียดเพิ่มเติม"
echo
echo -e "${GREEN}การทดสอบเสร็จสิ้น!${RESET}"
echo -e "${BLUE}สำหรับข้อมูลเพิ่มเติม โปรดดูที่ README-THAI-SUPPORT.md${RESET}"
