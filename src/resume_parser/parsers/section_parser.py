"""
Section-based resume parser - replacement for pattern-based extraction
"""

import re
from typing import Dict, List, Any
from datetime import datetime


def section_based_extraction(text: str) -> Dict[str, Any]:
    """Section-based extraction - identifies resume sections and extracts accordingly"""

    lines = [line.strip() for line in text.split('\n') if line.strip()]

    # First, identify all sections in the resume
    sections = identify_resume_sections(lines)

    # Extract data from identified sections
    return extract_from_sections(sections, text)


def identify_resume_sections(lines: List[str]) -> Dict[str, List[str]]:
    """Identify different sections in the resume"""

    sections = {
        'header': [],
        'contact': [],
        'summary': [],
        'skills': [],
        'experience': [],
        'education': [],
        'projects': [],
        'other': []
    }

    current_section = 'header'
    section_keywords = {
        'summary': ['summary', 'profile', 'objective', 'about'],
        'skills': ['skills', 'technical', 'technologies', 'tools', 'expertise', 'competencies'],
        'experience': ['experience', 'work', 'employment', 'career', 'professional', 'history'],
        'education': ['education', 'qualifications', 'academic', 'degrees', 'schooling'],
        'projects': ['projects', 'portfolio', 'work samples', 'achievements']
    }

    for i, line in enumerate(lines):
        line_lower = line.lower().strip()

        # Check if this line is a section header
        section_found = False
        for section_name, keywords in section_keywords.items():
            for keyword in keywords:
                # More flexible matching for section headers
                if (re.search(f'\\b{keyword}\\b', line_lower) and ':' in line) or \
                   re.match(f'^{keyword}s?:?\\s*$', line_lower) or \
                   line_lower == keyword:
                    current_section = section_name
                    section_found = True
                    break
            if section_found:
                break

        # Skip section headers themselves
        if section_found:
            continue

        # Add content to current section
        if line.strip():
            sections[current_section].append(line)

    return sections


def extract_from_sections(sections: Dict[str, List[str]], full_text: str) -> Dict[str, Any]:
    """Extract structured data from identified sections"""

    # Extract basic contact info from header and contact sections
    header_text = ' '.join(sections['header'] + sections['contact'])
    name = extract_name(sections['header'])
    email = extract_email(full_text)
    phone = extract_phone(full_text)
    location = extract_location(header_text)
    linkedin_url = extract_linkedin(full_text)

    # Extract skills from skills section
    skills = extract_skills_from_section(sections['skills'])

    # Extract experience from experience section
    experience_data = extract_experience_from_section(sections['experience'])

    # Extract education from education section
    education = extract_education_from_section(sections['education'])

    # Extract projects
    projects = extract_projects_from_section(sections['projects'])

    # Extract summary
    summary = extract_summary_from_section(sections['summary'])

    # Calculate total experience
    total_experience = calculate_total_experience(experience_data)

    # Determine current position
    current_position = extract_current_position(experience_data)

    # Generate recommendations
    recommended_roles = generate_role_recommendations(skills, current_position)
    best_role = determine_best_role(skills, current_position)

    return {
        "name": name,
        "email": email,
        "phone": phone,
        "skills": skills,
        "education": education,
        "experience": [exp['description'] for exp in experience_data],
        "projects": projects,
        "summary": summary,
        "recommended_roles": recommended_roles,
        "best_role": best_role,
        "linkedin_url": linkedin_url,
        "current_location": location,
        "current_position": current_position,
        "total_experience": total_experience
    }


def extract_name(header_lines: List[str]) -> str:
    """Extract name from header section"""
    for line in header_lines[:3]:  # Check first 3 lines
        # Skip lines with email, phone, or common keywords
        if re.search(r'@|\+?\d{10}|email|phone|resume', line.lower()):
            continue
        # Check if it looks like a name (2-4 words, mostly alphabetic)
        words = line.split()
        if 2 <= len(words) <= 4 and re.match(r'^[A-Za-z\s\.]+$', line.strip()):
            return line.strip()
    return "Unknown"


def extract_email(text: str) -> str:
    """Extract email from text"""
    pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    matches = re.findall(pattern, text)
    return matches[0] if matches else ""


def extract_phone(text: str) -> str:
    """Extract phone from text"""
    pattern = r'(\+91|91)?[\s-]?[6-9]\d{9}'
    matches = re.findall(pattern, text)
    return matches[0] if matches else ""


def extract_location(text: str) -> str:
    """Extract location from text"""
    # Look for city, state patterns
    location_pattern = r'([A-Za-z\s]+),\s*([A-Za-z\s]+)'
    matches = re.findall(location_pattern, text)
    if matches:
        return f"{matches[0][0]}, {matches[0][1]}"

    # Look for known cities
    cities = ['mumbai', 'bangalore', 'delhi', 'hyderabad', 'chennai', 'pune']
    for city in cities:
        if city in text.lower():
            return city.title()
    return ""


def extract_linkedin(text: str) -> str:
    """Extract LinkedIn URL"""
    pattern = r'linkedin\.com/in/([A-Za-z0-9_-]+)'
    matches = re.findall(pattern, text)
    return f"https://linkedin.com/in/{matches[0]}" if matches else ""


def extract_skills_from_section(skills_lines: List[str]) -> List[str]:
    """Extract skills from dedicated skills section"""
    skills = []
    for line in skills_lines:
        # Remove category labels first
        line = re.sub(r'^[^:]*:', '', line).strip()

        # Split by common delimiters
        line_skills = re.split(r'[,;•▪▫◦|\n\r]+', line)
        for skill in line_skills:
            skill = skill.strip().strip('•▪▫◦-*').strip()
            # Filter out very short strings and category headers
            if len(skill) > 2 and not re.match(r'^[:\-\s]+$', skill):
                # Skip category indicators
                if not any(cat in skill.lower() for cat in ['languages', 'frontend', 'backend', 'databases', 'cloud', 'tools']):
                    skills.append(skill)
    return list(dict.fromkeys(skills))  # Remove duplicates


def extract_experience_from_section(experience_lines: List[str]) -> List[Dict]:
    """Extract structured experience data"""
    experiences = []
    current_exp = {}

    for line in experience_lines:
        # Check if this is a job title line (contains job title keywords)
        job_keywords = ['engineer', 'developer', 'manager', 'analyst', 'architect', 'consultant']
        if any(keyword in line.lower() for keyword in job_keywords):
            if current_exp:
                experiences.append(current_exp)
            current_exp = {'description': line, 'details': []}
        else:
            # Add to current experience details
            if current_exp:
                current_exp['details'].append(line)
            else:
                # Standalone experience line
                current_exp = {'description': line, 'details': []}

    if current_exp:
        experiences.append(current_exp)

    return experiences


def extract_education_from_section(education_lines: List[str]) -> List[str]:
    """Extract education information"""
    education = []
    for line in education_lines:
        if len(line.strip()) > 5:  # Meaningful education entries
            education.append(line.strip())
    return education


def extract_projects_from_section(project_lines: List[str]) -> List[str]:
    """Extract project information"""
    projects = []
    for line in project_lines:
        if len(line.strip()) > 10:  # Meaningful project entries
            projects.append(line.strip())
    return projects


def extract_summary_from_section(summary_lines: List[str]) -> str:
    """Extract professional summary"""
    if summary_lines:
        return ' '.join(summary_lines)
    return ""


def calculate_total_experience(experience_data: List[Dict]) -> str:
    """Calculate total years of experience"""
    if not experience_data:
        return "Not specified"

    # Look for explicit year mentions
    try:
        all_text = ' '.join([
            exp.get('description', '') or ' '.join(exp.get('responsibilities', [])) or str(exp)
            for exp in experience_data if exp and isinstance(exp, dict)
        ])
    except Exception as e:
        print(f"Error processing experience data: {e}")
        return "0 years"
    exp_patterns = [
        r'(\d+(?:\.\d+)?)\+?\s*years?\s*(?:of\s*)?experience',
        r'(\d+(?:\.\d+)?)\+?\s*yrs?'
    ]

    for pattern in exp_patterns:
        matches = re.findall(pattern, all_text.lower())
        if matches:
            max_years = max([float(x) for x in matches])
            return f"{int(max_years) if max_years == int(max_years) else max_years} years"

    # Try to calculate from date ranges
    current_year = datetime.now().year
    earliest_year = None

    for exp in experience_data:
        text = exp['description']
        # Look for year patterns
        years = re.findall(r'(20\d{2})', text)
        for year in years:
            year_int = int(year)
            if 1990 <= year_int <= current_year:
                if earliest_year is None or year_int < earliest_year:
                    earliest_year = year_int

    if earliest_year:
        return f"{current_year - earliest_year} years"

    return "Not specified"


def extract_current_position(experience_data: List[Dict]) -> str:
    """Extract current job position"""
    if not experience_data:
        return ""

    # Look for current position indicators
    for exp in experience_data:
        desc = exp['description']
        if any(indicator in desc.lower() for indicator in ['present', 'current', 'now']):
            # Extract job title from the description
            title = desc.split(' at ')[0].strip() if ' at ' in desc else desc.strip()
            return title

    # Fallback to first experience entry
    if experience_data:
        desc = experience_data[0]['description']
        title = desc.split(' at ')[0].strip() if ' at ' in desc else desc.strip()
        return title

    return ""


def generate_role_recommendations(skills: List[str], current_position: str) -> List[str]:
    """Generate role recommendations based on skills"""
    recommendations = []
    skills_lower = [s.lower() for s in skills]

    # Role mapping based on skills
    if any(s in skills_lower for s in ['python', 'django', 'flask']):
        recommendations.append("Python Developer")
    if any(s in skills_lower for s in ['react', 'angular', 'vue']):
        recommendations.append("Frontend Developer")
    if any(s in skills_lower for s in ['java', 'spring']):
        recommendations.append("Java Developer")
    if any(s in skills_lower for s in ['aws', 'docker', 'kubernetes']):
        recommendations.append("DevOps Engineer")

    # Add current position if valid
    if current_position and len(current_position) > 5:
        recommendations.insert(0, current_position)

    # Default recommendations
    if not recommendations:
        recommendations = ["Software Engineer", "Full Stack Developer"]

    return list(dict.fromkeys(recommendations))[:5]


def determine_best_role(skills: List[str], current_position: str) -> str:
    """Determine the best fitting role"""
    if current_position and len(current_position.strip()) > 5:
        return current_position.strip()

    recommendations = generate_role_recommendations(skills, current_position)
    return recommendations[0] if recommendations else "Software Engineer"