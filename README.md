# 📊 Job Market Dashboard

A comprehensive data science job market analytics platform built with Streamlit.  Visualize job postings from multiple job boards that are updated daily, while matching opportunities against your resume using AI. Use the Salary Prediction Model to predict the salary of a role given details about it.

**Live Demo**: [https://data-science-jobs. streamlit.app/](https://data-science-jobs.streamlit.app/)

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.37.0+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## ✨ Features

### 🔍 Multi-Source Job Scraping
- Scrapes data science jobs from **Indeed**, **LinkedIn**, and **Glassdoor**
- Customizable search terms and locations
- Intelligent time-window sampling for trend analysis
- Results cached to AWS S3 for persistence

### 📈 Advanced Analytics Dashboard
- **Salary Analysis**: Visualize salary distributions and trends
- **Skills Intelligence**: Track most in-demand technical skills
- **Geographic Insights**: Explore job distribution by location
- **Trend Forecasting**: Linear regression-based salary predictions
- **Experience Analysis**: Breakdown by seniority level and education requirements

### AI-Powered Resume Matching
- Upload your resume (PDF format)
- Google Gemini AI extracts your skills and experience
- Intelligent matching algorithm scores jobs based on:
  - Technical skill overlap
  - Years of experience alignment
  - Education level match
  - Location preferences
- Get personalized job recommendations


## 📁 Project Structure

```
job-market-dashboard/
├── app.py                      # Main Streamlit application
├── scraper.py                  # Job scraping logic (JobSpy integration)
├── data_processor.py           # Data cleaning and feature extraction
├── resume_analyzer.py          # AI-powered resume parsing and matching
├── requirements.txt            # Python dependencies
├── requirements-actions.txt    # GitHub Actions specific dependencies
├── data/                       # Data directory
│   ├── raw_jobs.csv           # Scraped job listings
│   ├── processed_jobs.csv     # Cleaned and enriched data
│   ├── resume. pdf             # Your resume (upload via UI)
│   └── .cache/                # API response cache
├── scripts/                    # Automation scripts
├── utils/                      # Utility functions
│   └── config.py              # Environment configuration
├── docs/                       # Documentation
└── . github/                    # GitHub Actions workflows
```

## 🛠️ Technical Stack

| Component | Technology |
|-----------|-----------|
| **Frontend** | Streamlit |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Plotly |
| **Web Scraping** | JobSpy (Indeed/LinkedIn/Glassdoor) |
| **AI/ML** | Google Gemini API, Scikit-learn |
| **PDF Parsing** | PyMuPDF (fitz), PyPDF2 |
| **Cloud Storage** | AWS S3 (boto3) |
| **Configuration** | python-dotenv |

## 🔧 Advanced Configuration

}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.  For major changes: 

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [JobSpy](https://github.com/Bunsly/JobSpy) for job scraping capabilities
- [Streamlit](https://streamlit.io/) for the amazing dashboard framework
- [Google Gemini](https://deepmind.google/technologies/gemini/) for AI-powered resume analysis
- Job boards:  Indeed, LinkedIn, and Glassdoor for data sources

## 📧 Contact

**MShiyaji** - [@MShiyaji](https://github.com/MShiyaji)

Project Link: [https://github.com/MShiyaji/job-market-dashboard](https://github.com/MShiyaji/job-market-dashboard)

---

⭐ **Star this repository** if you find it helpful!

Built with ❤️ for job seekers and market analysts
