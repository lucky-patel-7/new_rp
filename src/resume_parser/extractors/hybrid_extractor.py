"""
Hybrid LLM + Regex Extractor
Combines pattern matching with AI intelligence for maximum accuracy
"""

import re
import json
from typing import List, Dict, Any, Optional
from ..clients.azure_openai import azure_client
import logging

logger = logging.getLogger(__name__)


class HybridExtractor:
    """Hybrid extraction using both LLM and regex for maximum accuracy"""

    def __init__(self):
        try:
            self.client = azure_client.get_sync_client()
            self.chat_deployment = azure_client.get_chat_deployment()
        except Exception as e:
            logger.warning(f"LLM client initialization failed: {e}")
            self.client = None

    def extract_contact_info(self, text: str) -> Dict[str, str]:
        """Hybrid contact extraction: regex first, then LLM validation/enhancement"""

        # Step 1: Regex-based extraction (fast)
        regex_results = self._regex_contact_extraction(text)

        # Step 2: LLM enhancement (accurate)
        if self.client:
            llm_results = self._llm_contact_enhancement(text, regex_results)
            return self._merge_contact_results(regex_results, llm_results)

        return regex_results

    def _regex_contact_extraction(self, text: str) -> Dict[str, str]:
        """Fast regex-based contact extraction"""

        # Email patterns
        email_patterns = [
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            r'[Ee]mail[:\s]*([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,})'
        ]

        # Phone patterns (enhanced)
        phone_patterns = [
            r'(\+91|91)?[\s-]?[6-9]\d{9}',
            r'\+\d{1,3}[\s-]?\d{3}[\s-]?\d{3}[\s-]?\d{4}',
            r'[Pp]hone[:\s]*(\+?\d[\s\-\(\)]*\d{9,})',
            r'[Mm]obile[:\s]*(\+?\d[\s\-\(\)]*\d{9,})'
        ]

        # LinkedIn patterns
        linkedin_patterns = [
            r'linkedin\.com/in/([A-Za-z0-9_-]+)',
            r'in\.linkedin\.com/([A-Za-z0-9_-]+)'
        ]

        results = {"email": "", "phone": "", "linkedin": ""}

        # Extract email
        for pattern in email_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                results["email"] = matches[0] if isinstance(matches[0], str) else matches[0][0]
                break

        # Extract phone
        for pattern in phone_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                phone = matches[0] if isinstance(matches[0], str) else matches[0][0]
                # Clean phone number
                results["phone"] = re.sub(r'[^\d+\-\s()]', '', phone).strip()
                break

        # Extract LinkedIn
        for pattern in linkedin_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                results["linkedin"] = f"https://linkedin.com/in/{matches[0]}"
                break

        return results

    def _llm_contact_enhancement(self, text: str, regex_results: Dict[str, str]) -> Dict[str, str]:
        """Use LLM to enhance and validate contact extraction"""

        try:
            prompt = f"""
Extract and validate contact information from this resume text.
Current regex results: {json.dumps(regex_results)}

Resume text:
{text[:2000]}  # Limit text for token efficiency

Please extract:
1. Email address (validate format)
2. Phone number (clean and format properly)
3. LinkedIn URL (complete URL)
4. Name (full name of the candidate)

Respond ONLY in JSON format:
{{
    "email": "email@domain.com or empty string",
    "phone": "+XX XXXXXXXXXX or empty string",
    "linkedin": "https://linkedin.com/in/username or empty string",
    "name": "Full Name or empty string"
}}
"""

            response = self.client.chat.completions.create(
                model=self.chat_deployment,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.1
            )

            llm_text = response.choices[0].message.content.strip()

            # Clean and parse JSON
            llm_text = re.sub(r"```json\s*([\s\S]*?)\s*```", r"\1", llm_text, flags=re.IGNORECASE)
            llm_text = re.sub(r"`{3,}", "", llm_text)

            return json.loads(llm_text)

        except Exception as e:
            logger.warning(f"LLM contact enhancement failed: {e}")
            return {}

    def _merge_contact_results(self, regex_results: Dict[str, str], llm_results: Dict[str, str]) -> Dict[str, str]:
        """Intelligently merge regex and LLM results"""

        merged = {}

        for key in ["email", "phone", "linkedin", "name"]:
            regex_val = regex_results.get(key, "")
            llm_val = llm_results.get(key, "")

            # Prefer LLM if it found something and regex didn't
            if llm_val and not regex_val:
                merged[key] = llm_val
            # Prefer regex if LLM didn't find anything
            elif regex_val and not llm_val:
                merged[key] = regex_val
            # If both found something, validate and choose best
            elif regex_val and llm_val:
                merged[key] = self._choose_better_result(key, regex_val, llm_val)
            else:
                merged[key] = ""

        return merged

    def _choose_better_result(self, field_type: str, regex_val: str, llm_val: str) -> str:
        """Choose the better result between regex and LLM"""

        if field_type == "email":
            # Validate email format
            email_pattern = r'^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}$'
            if re.match(email_pattern, llm_val, re.IGNORECASE):
                return llm_val
            elif re.match(email_pattern, regex_val, re.IGNORECASE):
                return regex_val

        elif field_type == "phone":
            # Prefer LLM if it's better formatted
            if len(llm_val.replace(' ', '').replace('-', '').replace('+', '')) >= 10:
                return llm_val
            return regex_val

        elif field_type == "linkedin":
            # Prefer complete URL
            if "linkedin.com/in/" in llm_val:
                return llm_val
            return regex_val

        elif field_type == "name":
            # Prefer LLM for name extraction (better at identifying names)
            if len(llm_val.split()) >= 2:
                return llm_val
            return regex_val

        return llm_val  # Default to LLM

    def extract_experience_hybrid(self, text: str, experience_list: List[str]) -> str:
        """Hybrid experience calculation: regex patterns + LLM intelligence"""

        # Step 1: Regex-based quick calculation
        regex_exp = self._regex_experience_calculation(text, experience_list)

        # Step 2: LLM-based intelligent calculation
        if self.client and experience_list:
            llm_exp = self._llm_experience_calculation(experience_list)
            return self._choose_better_experience(regex_exp, llm_exp)

        return regex_exp

    def _regex_experience_calculation(self, text: str, experience_list: List[str]) -> str:
        """Fast regex-based experience calculation"""

        # Look for explicit experience mentions
        exp_patterns = [
            r'(\d+(?:\.\d+)?)\+?\s*years?\s*(?:of\s*)?(?:work\s*)?experience',
            r'(?:total\s*)?experience.*?(\d+(?:\.\d+)?)\+?\s*years?',
            r'(\d+(?:\.\d+)?)\+?\s*yrs?\s*(?:of\s*)?(?:exp|experience)',
            r'over\s*(\d+(?:\.\d+)?)\+?\s*years?',
            r'more\s*than\s*(\d+(?:\.\d+)?)\+?\s*years?'
        ]

        for pattern in exp_patterns:
            matches = re.findall(pattern, text.lower())
            if matches:
                max_exp = max([float(x) for x in matches])
                if max_exp > 0:
                    return f"{int(max_exp) if max_exp == int(max_exp) else max_exp} years"

        return "Not specified"

    def _llm_experience_calculation(self, experience_list: List[str]) -> str:
        """Use LLM to intelligently calculate total experience"""

        try:
            experience_text = "\n".join(experience_list)

            prompt = f"""
Calculate the total years of professional work experience from this experience list.
Consider overlapping periods, gaps, and current date (2025).

Experience entries:
{experience_text}

Rules:
1. Count only professional work experience
2. Handle date ranges like "2020-2023", "2021-Present", "Jan 2019 - Mar 2021"
3. Don't double-count overlapping periods
4. Present/Current means December 2024
5. Return format: "X years Y months" or "X years" or "X months"

Respond with ONLY the calculated experience (e.g., "3 years 6 months"):
"""

            response = self.client.chat.completions.create(
                model=self.chat_deployment,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
                temperature=0.1
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.warning(f"LLM experience calculation failed: {e}")
            return "Not specified"

    def _choose_better_experience(self, regex_exp: str, llm_exp: str) -> str:
        """Choose the better experience calculation"""

        # If LLM provided a detailed calculation and regex didn't find anything
        if "years" in llm_exp.lower() and regex_exp == "Not specified":
            return llm_exp

        # If both found something, prefer LLM for complex calculations
        if "years" in llm_exp.lower() and "years" in regex_exp.lower():
            return llm_exp

        # Default to regex if LLM failed
        return regex_exp

    def extract_skills_hybrid(self, text: str, skills_section: List[str]) -> List[str]:
        """Hybrid skills extraction: pattern matching + LLM intelligence"""

        # Step 1: Regex-based pattern matching
        regex_skills = self._regex_skills_extraction(text, skills_section)

        # Step 2: LLM-based skills identification and cleaning
        if self.client:
            llm_skills = self._llm_skills_extraction(skills_section if skills_section else [text[:1000]])
            return self._merge_skills_results(regex_skills, llm_skills)

        return regex_skills

    def _regex_skills_extraction(self, text: str, skills_section: List[str]) -> List[str]:
        """Pattern-based skills extraction"""

        skills = []

        # Extract from dedicated skills section
        for line in skills_section:
            line = re.sub(r'^[^:]*:', '', line).strip()
            line_skills = re.split(r'[,;•▪▫◦|\n\r]+', line)
            for skill in line_skills:
                skill = skill.strip().strip('•▪▫◦-*').strip()
                if len(skill) > 2 and not re.match(r'^[:\-\s]+$', skill):
                    skills.append(skill)

        # If no skills section, use pattern matching on full text
        if not skills:
            skill_patterns = {
                # Programming Languages
                'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'go', 'rust', 'scala', 'r',
                # Web Technologies
                'react', 'angular', 'vue', 'node.js', 'express', 'django', 'flask', 'spring', 'html', 'css',
                # AI/ML Technologies
                'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'pandas', 'numpy', 'matplotlib', 'seaborn',
                'machine learning', 'deep learning', 'neural networks', 'artificial intelligence', 'nlp',
                'computer vision', 'opencv', 'xgboost', 'lightgbm', 'spark', 'kafka',
                # Cloud & DevOps
                'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'terraform', 'ansible',
                # Databases
                'postgresql', 'mysql', 'mongodb', 'redis', 'elasticsearch', 'sql server', 'oracle',
                # Tools & Others
                'git', 'jira', 'confluence', 'tableau', 'power bi', 'excel', 'jupyter', 'google analytics'
            }

            for skill in skill_patterns:
                if re.search(r'\b' + skill + r'\b', text.lower()):
                    skills.append(skill.title())

        return list(dict.fromkeys(skills))  # Remove duplicates

    def _llm_skills_extraction(self, skills_lines: List[str]) -> List[str]:
        """Use LLM to extract and clean skills"""

        try:
            skills_text = "\n".join(skills_lines)

            prompt = f"""
Extract technical skills from this resume content. Focus on:
- Programming languages
- Frameworks and libraries
- Tools and technologies
- Cloud platforms
- Databases

Skills content:
{skills_text}

Return ONLY a clean, comma-separated list of skills (no categories, no descriptions):
Example: Python, JavaScript, React, Docker, AWS, PostgreSQL
"""

            response = self.client.chat.completions.create(
                model=self.chat_deployment,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.1
            )

            skills_text = response.choices[0].message.content.strip()
            skills = [skill.strip() for skill in skills_text.split(',')]
            return [skill for skill in skills if len(skill) > 1]

        except Exception as e:
            logger.warning(f"LLM skills extraction failed: {e}")
            return []

    def _merge_skills_results(self, regex_skills: List[str], llm_skills: List[str]) -> List[str]:
        """Merge and deduplicate skills from both approaches"""

        # Combine both lists
        all_skills = regex_skills + llm_skills

        # Smart deduplication (case-insensitive)
        seen = set()
        merged_skills = []

        for skill in all_skills:
            skill_lower = skill.lower()
            if skill_lower not in seen:
                seen.add(skill_lower)
                merged_skills.append(skill)

        return merged_skills[:30]  # Limit to top 30 skills

    def extract_work_history_hybrid(self, text: str, experience_section: List[str]) -> List[Dict]:
        """Hybrid work history extraction: regex + LLM for comprehensive job details"""

        # Step 1: Regex-based structure extraction
        regex_jobs = self._regex_work_history_extraction(text, experience_section)

        # Step 2: LLM-based enhancement and validation
        if self.client and experience_section:
            llm_jobs = self._llm_work_history_extraction(experience_section)
            return self._merge_work_history_results(regex_jobs, llm_jobs)

        return regex_jobs

    def _regex_work_history_extraction(self, text: str, experience_section: List[str]) -> List[Dict]:
        """Regex-based work history extraction"""

        jobs = []
        current_job = {}

        for line in experience_section:
            line = line.strip()
            if not line:
                continue

            # Check for job title patterns
            job_patterns = [
                r'(senior|junior|lead|principal|staff|associate)?\s*(software|web|full.?stack|backend|frontend|data|mobile)\s*(engineer|developer|analyst|scientist)',
                r'(project|product|technical|program)\s*manager',
                r'(solution|software|system|cloud|security)\s*architect'
            ]

            # Check for company patterns (at/with company)
            company_pattern = r'\s+(?:at|with|@)\s+([A-Za-z0-9\s&.,]+?)(?:\s*[\(,]|$)'

            # Check for date patterns
            date_patterns = [
                r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|\d{1,2}/|\d{4})[^-]*?(?:-|to|–|—)\s*(?:present|current|now|\d{4})',
                r'\b(20\d{2})\s*[-–—]\s*(20\d{2}|present|current)\b'
            ]

            is_job_title = any(re.search(pattern, line, re.IGNORECASE) for pattern in job_patterns)

            if is_job_title:
                # Save previous job if exists
                if current_job:
                    jobs.append(current_job)

                # Start new job
                current_job = {
                    'title': line,
                    'company': '',
                    'duration': '',
                    'description': [],
                    'location': ''
                }

                # Extract company from same line
                company_match = re.search(company_pattern, line, re.IGNORECASE)
                if company_match:
                    current_job['company'] = company_match.group(1).strip()

            elif current_job:
                # Check if line contains date
                is_date_line = any(re.search(pattern, line, re.IGNORECASE) for pattern in date_patterns)
                if is_date_line:
                    current_job['duration'] = line
                elif 'company' in line.lower() or ' at ' in line.lower():
                    current_job['company'] = line
                else:
                    # Add to description
                    current_job['description'].append(line)

        # Add last job
        if current_job:
            jobs.append(current_job)

        return jobs

    def _llm_work_history_extraction(self, experience_section: List[str]) -> List[Dict]:
        """Use LLM to extract structured work history"""

        try:
            experience_text = "\n".join(experience_section)

            prompt = f"""
Extract detailed work history from this experience section. For each job, extract:
1. Job Title (e.g., "Senior Software Engineer")
2. Company Name (e.g., "TechCorp Solutions")
3. Duration (e.g., "March 2021 - Present")
4. Location (if mentioned)
5. Key Responsibilities (bullet points)

Experience section:
{experience_text}

Respond ONLY in JSON format as an array:
[
  {{
    "title": "Job Title",
    "company": "Company Name",
    "duration": "Start - End",
    "location": "City, Country",
    "responsibilities": ["Responsibility 1", "Responsibility 2"]
  }}
]
"""

            response = self.client.chat.completions.create(
                model=self.chat_deployment,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800,
                temperature=0.1
            )

            llm_text = response.choices[0].message.content.strip()

            # Clean and parse JSON
            llm_text = re.sub(r"```json\s*([\s\S]*?)\s*```", r"\1", llm_text, flags=re.IGNORECASE)
            llm_text = re.sub(r"`{3,}", "", llm_text)

            return json.loads(llm_text)

        except Exception as e:
            logger.warning(f"LLM work history extraction failed: {e}")
            return []

    def _merge_work_history_results(self, regex_jobs: List[Dict], llm_jobs: List[Dict]) -> List[Dict]:
        """Merge regex and LLM work history results"""

        # If LLM provided structured results, prefer those
        if llm_jobs and len(llm_jobs) > 0:
            return llm_jobs

        # Otherwise use regex results
        return regex_jobs

    def extract_projects_hybrid(self, text: str, projects_section: List[str]) -> List[Dict]:
        """Hybrid project extraction: regex + LLM for detailed project information"""

        # Step 1: Regex-based project identification
        regex_projects = self._regex_projects_extraction(text, projects_section)

        # Step 2: LLM-based enhancement
        if self.client and (projects_section or regex_projects):
            llm_projects = self._llm_projects_extraction(projects_section if projects_section else [text])
            return self._merge_projects_results(regex_projects, llm_projects)

        return regex_projects

    def _regex_projects_extraction(self, text: str, projects_section: List[str]) -> List[Dict]:
        """Regex-based project extraction"""

        projects = []

        # Look for project patterns
        project_patterns = [
            r'(project|built|developed|created|designed)\s*:?\s*([A-Za-z\s\-&]+)',
            r'([A-Za-z\s\-&]+)\s*[:]\s*(built|developed|created|designed)',
        ]

        search_text = "\n".join(projects_section) if projects_section else text

        for pattern in project_patterns:
            matches = re.findall(pattern, search_text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple) and len(match) >= 2:
                    project_name = match[1] if 'project' in match[0].lower() else match[0]
                    if len(project_name.strip()) > 5:
                        projects.append({
                            'name': project_name.strip(),
                            'description': '',
                            'technologies': []
                        })

        return projects[:10]  # Limit to top 10 projects

    def _llm_projects_extraction(self, projects_content: List[str]) -> List[Dict]:
        """Use LLM to extract detailed project information"""

        try:
            content_text = "\n".join(projects_content)

            prompt = f"""
Extract key projects from this resume content. For each project, identify:
1. Project Name
2. Brief Description
3. Technologies Used

Content:
{content_text[:1500]}

Respond ONLY in JSON format as an array:
[
  {{
    "name": "Project Name",
    "description": "Brief description of what the project does",
    "technologies": ["Tech1", "Tech2", "Tech3"]
  }}
]

Focus on real projects, not just technologies or job responsibilities.
"""

            response = self.client.chat.completions.create(
                model=self.chat_deployment,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.1
            )

            llm_text = response.choices[0].message.content.strip()

            # Clean and parse JSON
            llm_text = re.sub(r"```json\s*([\s\S]*?)\s*```", r"\1", llm_text, flags=re.IGNORECASE)
            llm_text = re.sub(r"`{3,}", "", llm_text)

            return json.loads(llm_text)

        except Exception as e:
            logger.warning(f"LLM projects extraction failed: {e}")
            return []

    def _merge_projects_results(self, regex_projects: List[Dict], llm_projects: List[Dict]) -> List[Dict]:
        """Merge regex and LLM project results"""

        # Prefer LLM results if available
        if llm_projects:
            return llm_projects

        return regex_projects

    def extract_education_hybrid(self, text: str, education_section: List[str]) -> List[Dict]:
        """Hybrid education extraction: regex + LLM for comprehensive education details"""

        # Step 1: Regex-based education extraction
        regex_education = self._regex_education_extraction(text, education_section)

        # Step 2: LLM-based enhancement
        if self.client and education_section:
            llm_education = self._llm_education_extraction(education_section)
            return self._merge_education_results(regex_education, llm_education)

        return regex_education

    def _regex_education_extraction(self, text: str, education_section: List[str]) -> List[Dict]:
        """Regex-based education extraction"""

        education = []
        search_text = "\n".join(education_section) if education_section else text

        # Education patterns
        degree_patterns = [
            r'(bachelor|master|b\.?tech|m\.?tech|b\.?e\.?|m\.?e\.?|bca|mca|mba|phd|diploma)\s*.*?(computer|software|information|engineering|science)',
            r'(computer science|software engineering|information technology|electrical engineering)',
            r'(university|college|institute)\s+of\s+(technology|engineering|science)'
        ]

        for pattern in degree_patterns:
            matches = re.finditer(pattern, search_text, re.IGNORECASE)
            for match in matches:
                education.append({
                    'degree': match.group(0),
                    'institution': '',
                    'year': '',
                    'field': ''
                })

        return education

    def _llm_education_extraction(self, education_section: List[str]) -> List[Dict]:
        """Use LLM to extract structured education information"""

        try:
            education_text = "\n".join(education_section)

            prompt = f"""
Extract education details from this section. For each education entry, extract:
1. Degree (e.g., "Bachelor of Technology", "B.Tech", "Master's")
2. Field of Study (e.g., "Computer Science", "Software Engineering")
3. Institution (e.g., "IIT Delhi", "University of Technology")
4. Year (e.g., "2015-2019", "2019")

Education section:
{education_text}

Respond ONLY in JSON format as an array:
[
  {{
    "degree": "Degree Type",
    "field": "Field of Study",
    "institution": "Institution Name",
    "year": "Year or Range"
  }}
]
"""

            response = self.client.chat.completions.create(
                model=self.chat_deployment,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
                temperature=0.1
            )

            llm_text = response.choices[0].message.content.strip()

            # Clean and parse JSON
            llm_text = re.sub(r"```json\s*([\s\S]*?)\s*```", r"\1", llm_text, flags=re.IGNORECASE)
            llm_text = re.sub(r"`{3,}", "", llm_text)

            return json.loads(llm_text)

        except Exception as e:
            logger.warning(f"LLM education extraction failed: {e}")
            return []

    def _merge_education_results(self, regex_education: List[Dict], llm_education: List[Dict]) -> List[Dict]:
        """Merge regex and LLM education results"""

        # Prefer LLM results if available
        if llm_education:
            return llm_education

        return regex_education

    def extract_current_employment_hybrid(self, text: str, experience_data: List[Dict]) -> Dict[str, str]:
        """Extract current company and position details"""

        try:
            # Look for "Present", "Current" indicators in experience
            current_job = {"company": "", "position": "", "duration": "", "location": ""}

            for job in experience_data:
                if isinstance(job, dict):
                    duration = job.get('duration', '')
                    if any(indicator in duration.lower() for indicator in ['present', 'current', 'now']):
                        current_job = {
                            "company": job.get('company', ''),
                            "position": job.get('title', ''),
                            "duration": duration,
                            "location": job.get('location', '')
                        }
                        break

            # If no current job found in structured data, use LLM
            if not current_job['company'] and self.client:
                return self._llm_current_employment_extraction(text)

            return current_job

        except Exception as e:
            logger.warning(f"Current employment extraction failed: {e}")
            return {"company": "", "position": "", "duration": "", "location": ""}

    def _llm_current_employment_extraction(self, text: str) -> Dict[str, str]:
        """Use LLM to extract current employment details"""

        try:
            prompt = f"""
Extract current employment details from this resume:

Resume text:
{text[:1500]}

Find the candidate's current job and extract:
1. Current Company Name
2. Current Position/Job Title
3. Employment Duration
4. Work Location

Respond ONLY in JSON format:
{{
    "company": "Current Company Name",
    "position": "Current Job Title",
    "duration": "Start Date - Present",
    "location": "Work Location"
}}

If no current job is found, return empty strings.
"""

            response = self.client.chat.completions.create(
                model=self.chat_deployment,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.1
            )

            llm_text = response.choices[0].message.content.strip()

            # Clean and parse JSON
            llm_text = re.sub(r"```json\s*([\s\S]*?)\s*```", r"\1", llm_text, flags=re.IGNORECASE)
            llm_text = re.sub(r"`{3,}", "", llm_text)

            return json.loads(llm_text)

        except Exception as e:
            logger.warning(f"LLM current employment extraction failed: {e}")
            return {"company": "", "position": "", "duration": "", "location": ""}