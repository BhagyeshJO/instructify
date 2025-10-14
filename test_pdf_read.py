import pdfplumber

# 👇 change this to your actual file name
path = r"C:\Users\Kiran Joshi\Downloads\ENVIRONMENTAL_HEALTH_and_SAFETY_POLICY_eb75053f4b.pdf"

with pdfplumber.open(path) as pdf:
    first_page = pdf.pages[0]
    text = first_page.extract_text()
    print("Extracted text:\n")
    print(text)
