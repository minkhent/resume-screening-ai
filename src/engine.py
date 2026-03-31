import spacy
from sentence_transformers import SentenceTransformer, util
import numpy as np
import re

class ResumeEngine:
    def __init__(self):
        # Semantic similarity model
        self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
        # Tokenization and text preprocessing
        self.nlp = spacy.load("en_core_web_md")

    # ---------------- CLEANING ----------------
    def preprocess_text(self, text):
        return text.lower().strip()

    # ---------------- EXPERIENCE SIGNAL ----------------
    def extract_experience_score(self, text):
        """
        Detect seniority via patterns like:
        - "5+ years"
        - "led", "architected", "owned"
        """
        score = 0.0

        # Years of experience
        years = re.findall(r'(\d+)\+?\s+years', text)
        if years:
            max_years = max([int(y) for y in years])
            score += min(max_years / 10, 1.0)  # normalize to 1.0 max

        # Senior verbs
        senior_keywords = [
            "led", "owned", "architected", "designed",
            "scaled", "optimized", "built", "managed"
        ]
        for word in senior_keywords:
            if word in text:
                score += 0.1

        return min(score, 1.0)

    # ---------------- SKILL MATCHING ----------------
    def compute_skill_score(self, resume_text, job_json):
        """
        Use skill alias mapping for normalization, then semantic similarity.
        """
        found = []
        missing = []

        # Flatten all target skills + alias mapping
        targets = job_json.get("core_competencies", []) + job_json.get("technical_stack", [])
        skill_alias = job_json.get("skill_alias_mapping", {})

        # Create normalized mapping: alias -> canonical skill
        normalized_map = {}
        for canonical, aliases in skill_alias.items():
            for alias in aliases:
                normalized_map[alias.lower()] = canonical

        # Lowercase resume
        resume_lower = resume_text.lower()
        resume_embedding = self.semantic_model.encode(resume_lower)

        for skill in targets:
            skill_lower = skill.lower()
            canonical_skill = normalized_map.get(skill_lower, skill_lower)

            # Direct keyword match in resume
            if any(alias in resume_lower for alias in [skill_lower] + skill_alias.get(canonical_skill, [])):
                found.append(skill)
                continue

            # Semantic similarity fallback
            skill_embedding = self.semantic_model.encode(skill_lower)
            sim = util.cos_sim(resume_embedding, skill_embedding).item()
            if sim > 0.55:  # tuned threshold
                found.append(f"{skill} (semantic)")
            else:
                missing.append(skill)

        score = len(found) / len(targets) if targets else 0
        return score, found, missing

    # ---------------- CONTEXT SCORE ----------------
    def compute_context_score(self, resume_text, job_json):
        context_blob = " ".join(
            job_json.get("core_competencies", []) +
            job_json.get("experience_benchmarks", []) +
            [job_json.get("industry_standard_summary", "")]
        )
        emb = self.semantic_model.encode([resume_text, context_blob])
        return util.cos_sim(emb[0], emb[1]).item()

    # ---------------- MAIN PIPELINE ----------------
    def analyze(self, resume_text, job_json):
        resume_text = self.preprocess_text(resume_text)

        # --- SKILL MATCH ---
        skill_score, found, missing = self.compute_skill_score(resume_text, job_json)

        # --- EXPERIENCE ---
        exp_score = self.extract_experience_score(resume_text)

        # --- CONTEXT ---
        context_score = self.compute_context_score(resume_text, job_json)

        # --- FINAL WEIGHTED SCORE ---
        final_score = (
            skill_score * 0.35 +
            exp_score * 0.20 +
            context_score * 0.25 +
            (1 if found else 0) * 0.20  # keyword presence boost
        )

        return {
            "match_percentage": round(final_score * 100, 2),
            "context_integrity": round(context_score * 100, 2),
            "experience_score": round(exp_score * 100, 2),
            "skill_coverage": round(skill_score * 100, 2),
            "found_skills": found[:15],  # UI control
            "missing_skills": missing[:15]
        }

