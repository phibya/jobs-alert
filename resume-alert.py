# %%
# Import necessary libraries
import time
import os
import argparse
import json
from typing import Dict, List, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import ollama
import pymupdf4llm

from api.JobPosting import NewestJobPostingAPI
from api.PushOver import PushOverAPI
from querydict.parser import QueryEngine


# %%
class ResumeJobMatcher:
    def __init__(self, device: str = "cpu", llm_model: str = "qwen2.5:7b", embedding_model: str = "BAAI/bge-m3"):
        """
        Initialize the resume-job matcher with specified device and models

        Args:
            device: Device for embedding model (cpu, cuda:0, mps, etc.)
            llm_model: Ollama model name (e.g., qwen2.5:7b, llama3.1:8b)
            embedding_model: HuggingFace embedding model name
        """
        self.device = device
        self.llm_model = llm_model
        self.embedding_model_name = embedding_model
        self.setup_device()

        # Initialize embedding model
        print(f"Loading {embedding_model} embedding model on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model)
        self.embedding_model = AutoModel.from_pretrained(embedding_model).to(self.device)
        self.embedding_model.eval()

        # Get embedding dimension
        self.embedding_dim = self.embedding_model.config.hidden_size

        # Initialize Ollama client
        print(f"Initializing Ollama with model: {llm_model}...")
        self.ollama_client = ollama.Client()

        # Check if model is available, pull if needed
        try:
            self.ollama_client.show(llm_model)
            print(f"✅ Model {llm_model} is available")
        except ollama.ResponseError:
            print(f"Pulling model {llm_model}...")
            self.ollama_client.pull(llm_model)
            print(f"✅ Model {llm_model} pulled successfully")

        # Section weights for final score calculation
        self.section_weights = {
            "skills": 0.50,
            "experience": 0.35,
            "education": 0.15
        }

        # Resume embeddings (computed once)
        self.resume_embeddings = None
        self.resume_sections = None

    def setup_device(self):
        """Setup device for PyTorch operations"""
        if self.device == "mps" and not torch.backends.mps.is_available():
            print("MPS not available, falling back to CPU")
            self.device = "cpu"
        elif "cuda" in self.device and not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            self.device = "cpu"

    def encode_text(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts using BGE-M3 model

        Args:
            texts: List of texts to encode

        Returns:
            Numpy array of embeddings
        """
        # Tokenize
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors='pt',
            max_length=512
        )

        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate embeddings
        with torch.no_grad():
            outputs = self.embedding_model(**inputs)
            # Mean pooling
            embeddings = outputs.last_hidden_state.mean(dim=1)
            # Normalize
            embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings.cpu().numpy()

    def call_ollama(self, prompt: str, max_tokens: int = 1024) -> str:
        """
        Call Ollama API with the given prompt

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate

        Returns:
            Generated response
        """
        try:
            response = self.ollama_client.generate(
                model=self.llm_model,
                prompt=prompt,
                options={
                    'temperature': 0.1,
                    'top_p': 0.9,
                    'num_predict': max_tokens,
                }
            )
            return response['response'].strip()
        except Exception as e:
            print(f"Error calling Ollama: {e}")
            return ""

    def parse_resume(self, resume_path: str) -> str:
        """
        Parse resume from PDF file using pymupdf4llm

        Args:
            resume_path: Path to the resume PDF file

        Returns:
            Extracted text from the resume
        """
        try:
            # Extract text from PDF
            resume_text = pymupdf4llm.to_markdown(resume_path)
            return resume_text
        except Exception as e:
            print(f"Error parsing resume: {e}")
            return ""

    def structure_resume_with_llm(self, resume_text: str) -> Dict[str, str]:
        """
        Use LLM to structure the resume text into sections (one section at a time)

        Args:
            resume_text: Raw resume text

        Returns:
            Dictionary with structured resume sections
        """
        sections = {}

        # Define section extraction prompts
        section_prompts = {
            "skills": """Extract and infer ALL skills from this entire resume. Include:
- Technical skills: programming languages, frameworks, tools, platforms
- Soft skills: inferred from activities (teaching = communication/mentorship, leading projects = leadership, etc.)
- Domain expertise: industry knowledge, methodologies, certifications
- Languages: if mentioned

Examples of skill inference:
- Teaching/Training → Communication, Mentorship, Public Speaking
- Leading projects → Leadership, Project Management
- Research → Analytical Thinking, Problem Solving
- Volunteer work → Teamwork, Community Engagement
- Presentations → Public Speaking, Communication

Return as comma-separated list.""",

            "experience": """Extract work experience and summarize by:
- Industry/Field (e.g., fintech, healthcare, e-commerce, SaaS, consulting)
- General project types (e.g., web applications, mobile apps, data pipelines, ML models, infrastructure)
- Scale of work (e.g., startup environment, enterprise systems, consumer products)
- Role type (e.g., individual contributor, team lead, architect, consultant)

Format as bullet points with • for each role focusing on WHAT they built/worked on, not WHERE:

Example:
• Senior Engineer: Fintech applications, payment processing systems, regulatory compliance tools
• Lead Developer: E-commerce platforms, inventory management, customer-facing web applications
• Data Scientist: Healthcare ML models, patient outcome prediction, clinical data analysis""",

            "education": """Extract educational background. Include only:
- Degree type (Bachelor's, Master's, PhD, etc.)
- Major/Field of study
- Graduation year (if available)
- Relevant coursework, certifications, or academic projects
- Any teaching/research experience (but don't repeat company names)

DO NOT include university names, locations, or specific school details. Format as bullet points with •"""
        }

        for section_name, section_prompt in section_prompts.items():
            print(f"Extracting {section_name} section...")

            prompt = f"""{section_prompt}

Resume:
{resume_text}

{section_name.title()}:"""

            try:
                response = self.call_ollama(prompt, max_tokens=512)
                # Clean up the response
                section_content = response.strip()

                # Basic validation - ensure we got something meaningful
                if section_content and len(section_content) > 10:
                    sections[section_name] = section_content
                else:
                    sections[section_name] = self._get_fallback_section(section_name, resume_text)

            except Exception as e:
                print(f"Error extracting {section_name}: {e}")
                sections[section_name] = self._get_fallback_section(section_name, resume_text)

        return sections

    def _get_fallback_section(self, section_name: str, resume_text: str) -> str:
        """Generate fallback content for a section if LLM extraction fails"""
        fallbacks = {
            "skills": "Problem-solving, Communication, Teamwork, Adaptability, Technical Analysis, Software Development",
            "experience": "• Professional role: Technology industry, software development projects, collaborative team environment",
            "education": "• Degree in relevant field\n• Technical coursework and training"
        }
        return fallbacks.get(section_name, "Information available in resume")

    def initialize_resume_embeddings(self, resume_text: str):
        """
        Structure resume and generate embeddings for each section (called once)

        Args:
            resume_text: Raw resume text
        """
        print("Structuring resume into sections...")
        self.resume_sections = self.structure_resume_with_llm(resume_text)

        # Print extracted resume sections for user review
        print("\n" + "=" * 60)
        print("📄 EXTRACTED RESUME SECTIONS")
        print("=" * 60)

        for section_name, section_content in self.resume_sections.items():
            print(f"\n🔹 {section_name.upper()}:")
            print("-" * 40)
            print(section_content)
            print()

        print("=" * 60)
        print("Resume parsing complete! Please review the extracted sections above.")
        print("=" * 60)

        print("\nGenerating resume section embeddings...")
        self.resume_embeddings = {}

        # Prepare texts for batch encoding
        section_texts = []
        section_names = []

        for section_name, section_text in self.resume_sections.items():
            if section_text and section_text.strip():
                section_texts.append(section_text)
                section_names.append(section_name)

        # Batch encode all sections at once for efficiency
        if section_texts:
            embeddings = self.encode_text(section_texts)
            for i, section_name in enumerate(section_names):
                self.resume_embeddings[section_name] = embeddings[i]

        # Add zero embeddings for missing sections
        for section_name in self.section_weights.keys():
            if section_name not in self.resume_embeddings:
                self.resume_embeddings[section_name] = np.zeros(self.embedding_dim)

        print(f"✅ Resume embeddings generated for sections: {list(self.resume_embeddings.keys())}")

    def generate_ideal_section_from_job(self, job_posting: Dict, section_name: str) -> str:
        """
        Generate ideal content for a specific section based on job posting

        Args:
            job_posting: Job posting dictionary
            section_name: Section to generate (summary, skills, experience, education)

        Returns:
            Generated ideal section content
        """
        title = job_posting.get("title", "")
        company = job_posting.get("hiringOrganization", {}).get("name", "")
        description = job_posting.get("description", "")
        requirements = job_posting.get("qualifications", "")

        section_prompts = {
            "skills": f"List the key technical and soft skills (comma-separated) needed for this {title} position. Focus on skills mentioned in the requirements and job description.",
            "experience": f"Describe the ideal industry background and project types for this {title} role. Focus on relevant domains, types of systems/applications, and work environments.",
            "education": f"Describe the ideal educational background for this {title} position. Include degree level, relevant fields of study, and certifications."
        }

        prompt = f"""You are creating ideal candidate profiles. Generate ONLY the requested section content, keep it concise and relevant.

Job: {title} at {company}
Description: {description[:300]}...
Requirements: {requirements[:300]}...

Task: {section_prompts.get(section_name, "Generate relevant content for this section.")}

Response:"""

        try:
            ideal_content = self.call_ollama(prompt, max_tokens=512)
            return ideal_content if ideal_content else f"Ideal {section_name} for {title} position"
        except Exception as e:
            print(f"Error generating ideal {section_name}: {e}")
            return f"Ideal {section_name} for {title} position"

    def calculate_job_match_score(self, job_posting: Dict) -> Dict:
        """
        Calculate match score for a job posting against pre-computed resume embeddings

        Args:
            job_posting: Job posting dictionary

        Returns:
            Dictionary with matching results
        """
        if not self.resume_embeddings:
            raise ValueError("Resume embeddings not initialized. Call initialize_resume_embeddings first.")

        section_scores = {}

        # Generate ideal content and calculate similarity for each section
        for section_name in self.section_weights.keys():
            if section_name in self.resume_embeddings:
                # Generate ideal section content
                ideal_content = self.generate_ideal_section_from_job(job_posting, section_name)

                # Generate embedding for ideal content
                ideal_embedding = self.encode_text([ideal_content])[0]

                # Calculate similarity
                similarity = cosine_similarity(
                    [self.resume_embeddings[section_name]],
                    [ideal_embedding]
                )[0][0]

                # Convert to 0-100 scale
                score = (similarity + 1) * 50
                section_scores[section_name] = min(100, max(0, score))
            else:
                section_scores[section_name] = 0

        # Calculate weighted overall score
        overall_score = sum(
            section_scores.get(section, 0) * weight
            for section, weight in self.section_weights.items()
        )

        return {
            "job_title": job_posting.get("title", ""),
            "company": job_posting.get("hiringOrganization", {}).get("name", ""),
            "overall_score": round(overall_score, 2),
            "section_scores": section_scores
        }


# %%
# Set up variables and argument parsing
parser = argparse.ArgumentParser("python resume-job-matcher.py")
parser.add_argument("--pushover_user_key", type=str, help="Pushover user key")
parser.add_argument("--pushover_api_token", type=str, help="Pushover API token")
parser.add_argument("--notification_interval", type=int, default=3600, help="Interval in seconds to send notifications")
parser.add_argument("--query", type=str, help="Keywords to match in job postings (optional)")
parser.add_argument("--resume_path", type=str, required=True, help="Path to resume PDF file")
parser.add_argument("--device", type=str, default="cpu", help="Device for models (cpu, cuda:0, mps, etc.)")
parser.add_argument("--embedding_model", type=str, default="BAAI/bge-m3", help="HuggingFace embedding model name")
parser.add_argument("--llm_model", type=str, default="qwen2.5:7b",
                    help="Ollama model name (e.g., qwen2.5:7b, llama3.1:8b)")
parser.add_argument("--min_score", type=float, default=70.0, help="Minimum matching score to trigger notification")
args = parser.parse_args()

# Update PUSHOVER_CONFIG with command line arguments
PUSHOVER_CONFIG = {
    "user_key": args.pushover_user_key,
    "api_token": args.pushover_api_token
}

NOTIFICATION_INTERVAL = args.notification_interval
QUERY = args.query
RESUME_PATH = args.resume_path
DEVICE = args.device
EMBEDDING_MODEL = args.embedding_model
LLM_MODEL = args.llm_model
MIN_SCORE = args.min_score

# Validate resume path
if not os.path.exists(RESUME_PATH):
    raise ValueError(f"Resume file not found: {RESUME_PATH}")

# Initialize keyword matching if query is provided
qe = None
if QUERY:
    qe = QueryEngine(QUERY)
    print(f"Keyword query: {QUERY}")

print(f"Resume path: {RESUME_PATH}")
print(f"Device: {DEVICE}")
print(f"Embedding model: {EMBEDDING_MODEL}")
print(f"LLM model: {LLM_MODEL}")
print(f"Minimum score threshold: {MIN_SCORE}")


# %%
def match_func(job_posting: dict, resume_matcher: ResumeJobMatcher) -> tuple:
    """
    Enhanced matching function: first applies keyword filter, then resume matching

    Args:
        job_posting: Job posting dictionary
        resume_matcher: ResumeJobMatcher instance

    Returns:
        Tuple of (is_match, score, match_details)
    """
    # First apply keyword filter if query is provided
    if qe and not qe.match(job_posting):
        return False, 0.0, None

    # If keyword filter passes (or no keyword filter), perform resume matching
    match_result = resume_matcher.calculate_job_match_score(job_posting)
    score = match_result["overall_score"]

    # Check if score meets minimum threshold
    is_match = score >= MIN_SCORE

    return is_match, score, match_result


# %%
def main():
    # Initialize the matcher
    print("Initializing Resume-Job Matcher...")
    resume_matcher = ResumeJobMatcher(device=DEVICE, llm_model=LLM_MODEL, embedding_model=EMBEDDING_MODEL)

    # Parse resume once at startup and generate embeddings
    print("Parsing resume...")
    resume_text = resume_matcher.parse_resume(RESUME_PATH)
    if not resume_text:
        raise ValueError("Failed to parse resume. Please check the file path and format.")

    print("Generating resume embeddings (one-time setup)...")
    resume_matcher.initialize_resume_embeddings(resume_text)
    print("✅ Resume analysis complete!")

    # Initialize the API client
    api = NewestJobPostingAPI()
    pushover_api = PushOverAPI(
        api_token=PUSHOVER_CONFIG["api_token"],
        user_key=PUSHOVER_CONFIG["user_key"]
    )

    # Initialize variables for notification
    last_notification_sent_time = time.time()
    jobs_to_notify = []

    # Loop to continuously check for new job postings
    while True:
        print(f"\nChecking for new job postings at {time.strftime('%Y-%m-%d %H:%M:%S')}...")

        try:
            # Fetch the latest job postings
            job_postings = api.get()
            print(f"Fetched {len(job_postings)} job postings.")

            # Filter job postings: first keyword filter, then LLM matching
            matched_jobs = 0
            keyword_matched_jobs = 0

            for job in job_postings:
                # Check keyword filter first
                keyword_match = not qe or qe.match(job)

                if keyword_match:
                    keyword_matched_jobs += 1

                    # Perform LLM matching for keyword-matched jobs
                    is_match, score, match_details = match_func(job, resume_matcher)

                    # Print score for all keyword-matched jobs
                    title = job.get('title', 'Unknown Title')
                    company = job.get('hiringOrganization', {}).get('name', 'Unknown Company')
                    url = job.get('linkedInUrl', 'No URL')

                    if is_match:
                        matched_jobs += 1
                        section_scores = match_details.get("section_scores", {})
                        skills_score = section_scores.get("skills", 0)
                        exp_score = section_scores.get("experience", 0)

                        print(f"✅ MATCH | Score: {score:.1f}% | {title} at {company}")
                        print(f"   Skills: {skills_score:.1f}% | Experience: {exp_score:.1f}% | URL: {url}")

                        # Add to notification list
                        job["match_score"] = score
                        job["match_details"] = match_details
                        jobs_to_notify.append(job)
                    else:
                        # Below threshold but passed keyword filter
                        print(f"❌ LOW   | Score: {score:.1f}% | {title} at {company}")
                        print(f"   URL: {url}")

            print(f"\n📊 SUMMARY:")
            print(f"   Total jobs fetched: {len(job_postings)}")
            if qe:
                print(f"   Jobs matching keywords: {keyword_matched_jobs}")
            print(f"   Jobs above threshold ({MIN_SCORE}%): {matched_jobs}")
            print(f"   Jobs queued for notification: {len(jobs_to_notify)}")
            print(f"-" * 60)

        except Exception as e:
            print(f"Error processing job postings: {e}")
            time.sleep(30)
            continue

        # Check if enough time has passed to send a notification
        current_time = time.time()
        if len(jobs_to_notify) > 0 and (current_time - last_notification_sent_time) >= NOTIFICATION_INTERVAL:
            # Send notification
            print(f"\nSending notification for {len(jobs_to_notify)} matching jobs...")

            # Sort jobs by score (highest first)
            jobs_to_notify.sort(key=lambda x: x.get("match_score", 0), reverse=True)

            message = f"""🎯 {len(jobs_to_notify)} high-matching job postings found!

{chr(10).join(
                f"📋 {job.get('title')} at {job.get('hiringOrganization', {}).get('name')}\n"
                f"📊 Overall: {job.get('match_score', 0):.1f}% | Skills: {job.get('match_details', {}).get('section_scores', {}).get('skills', 0):.1f}% | Exp: {job.get('match_details', {}).get('section_scores', {}).get('experience', 0):.1f}%\n"
                f"🔗 {job.get('linkedInUrl', 'No URL')}\n"
                for job in jobs_to_notify[:5]  # Limit to top 5 to avoid message length issues
            )}

{f'... and {len(jobs_to_notify) - 5} more jobs!' if len(jobs_to_notify) > 5 else ''}
"""

            # Send the notification using PushOver
            response = pushover_api.send_message(
                message=message,
                title="🚀 High-Match Job Alerts",
                priority=1
            )

            # Log the response from PushOver
            if response.get("status") != 1:
                print(f"Failed to send notification: {response.get('errors', 'Unknown error')}")
            else:
                print(f"✅ Notification sent at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(current_time))}")

                # Update the last notification sent time
                last_notification_sent_time = current_time

                # Clear the jobs to notify list after sending
                jobs_to_notify.clear()

        # Wait before checking again
        time.sleep(60)  # Check every minute


if __name__ == "__main__":
    main()