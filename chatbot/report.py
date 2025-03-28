import pdfkit
import tempfile

# Generate Medical Report as PDF
def generate_pdf(image_description, diagnosis, medicine, classification):
    html_content = f"""
    <h2>Lung Cancer Detection Report</h2>
    <p><b>AI Image Analysis:</b> {image_description}</p>
    <h3>AI Classification</h3>
    <p><b>Prediction:</b> {classification}</p>
    <h3>Doctor's Notes</h3>
    <p>{diagnosis}</p>
    <h3>Recommended Medicines</h3>
    <p>{medicine}</p>
    """

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
        pdfkit.from_string(html_content, tmp_pdf.name)
        return tmp_pdf.name
