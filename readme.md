See:
https://medium.com/data-science-collective/when-an-ai-tool-i-built-evaluated-my-resume-i-learned-what-100-rejections-never-taught-me-8e8eea1f3d8f

# Set up the environment

1. Install dependencies:
pip install streamlit openai python-dotenv PyPDF2 docx2txt scikit-learn

2. Add your OpenAI key to a .env file
echo "OPENAI_API_KEY=your-api-key-here" > .env

# Deploy and test
streamlit run ats_app.py
