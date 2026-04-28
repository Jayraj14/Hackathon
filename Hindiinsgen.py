from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.pagesizes import A4

# Register Hindi font
pdfmetrics.registerFont(TTFont('NotoSansDevanagari', 'NotoSansDevanagari-Regular.ttf'))

# Create PDF
file_name = "insurance_policy_hindi.pdf"
c = canvas.Canvas(file_name, pagesize=A4)

# Set font (important for Hindi text)
c.setFont("NotoSansDevanagari", 14)

# Sample Hindi insurance policy text
text = [
    "बीमा पॉलिसी दस्तावेज़",
    "",
    "पॉलिसीधारक का नाम: राम कुमार",
    "पॉलिसी नंबर: INS123456789",
    "बीमा राशि: ₹5,00,000",
    "पॉलिसी अवधि: 01/01/2026 से 01/01/2031",
    "",
    "नियम और शर्तें:",
    "1. यह पॉलिसी स्वास्थ्य और जीवन बीमा कवरेज प्रदान करती है।",
    "2. प्रीमियम का भुगतान समय पर करना आवश्यक है।",
    "3. किसी भी दावे के लिए आवश्यक दस्तावेज जमा करने होंगे।"
]

# Write text line by line
y = 800
for line in text:
    c.drawString(50, y, line)
    y -= 25

# Save PDF
c.save()

print("PDF successfully created:", file_name)