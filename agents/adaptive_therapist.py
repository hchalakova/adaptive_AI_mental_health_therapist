"""
Adaptive Therapist Agent (Component 3)

Replaces the static MentalHealthAssistant. Instead of rigidly stepping through
questionnaire questions, this agent:
1. Analyzes the patient's response for resistance (via ResistanceDetector)
2. Selects a therapeutic strategy (via PolicySelector)
3. Generates a response that follows the strategy while pursuing session goals

The agent maintains a conversation state tracking:
- Which symptom areas have been explored
- Which questionnaire questions have been covered (directly or indirectly)
- The current resistance trend
- What information still needs to be gathered
"""

import os
import re
import logging
from typing import List, Dict, Any, Optional, Tuple

from utils.resistance_detector import ResistanceDetector, ResistanceEstimate, ResistanceType
from utils.policy_selector import PolicySelector, PolicyMode, PolicyDecision

logger = logging.getLogger(__name__)


class ConversationState:
    """Tracks the state of the therapeutic conversation."""

    def __init__(self, questions: List[str]):
        self.questions = questions
        self.total_questions = len(questions)
        # Track which questions have been covered (asked or info gathered)
        self.questions_covered = [False] * len(questions)
        self.questions_attempted = [False] * len(questions)
        # Track how many times each question has been attempted
        self.attempt_counts = [0] * len(questions)
        self.max_attempts_per_question = 2  # Move on after 2 failed attempts
        # Track patient responses per question
        self.responses: Dict[int, str] = {}
        # Current question index for sequential fallback
        self.next_question_idx = 0
        # Symptom areas explored (free-form notes)
        self.symptoms_explored: List[str] = []
        # Number of total exchanges
        self.turn_count = 0
        # Consecutive resistance turns
        self.consecutive_resistance_turns = 0
        # Track if we're in "open conversation" mode (paused questionnaire)
        self.open_conversation_mode = False

    def mark_question_attempted(self, idx: int):
        """Mark that we attempted to get info for question idx."""
        if 0 <= idx < len(self.questions_attempted):
            self.questions_attempted[idx] = True
            self.attempt_counts[idx] += 1

    def mark_question_covered(self, idx: int, response: str):
        """Mark that question idx has been adequately answered."""
        if 0 <= idx < len(self.questions_covered):
            self.questions_covered[idx] = True
            self.responses[idx] = response

    def mark_question_skipped(self, idx: int):
        """Mark a question as skipped due to too many failed attempts."""
        if 0 <= idx < len(self.questions_covered):
            self.questions_covered[idx] = True
            self.responses[idx] = "[SKIPPED — patient refused to engage on this topic]"

    def get_next_uncovered_question(self) -> Optional[Tuple[int, str]]:
        """
        Get the next question that hasn't been covered yet.
        Skips questions that have been attempted too many times without success.
        """
        for i in range(self.total_questions):
            if self.questions_covered[i]:
                continue
            # If we've tried this question too many times, skip it
            if self.attempt_counts[i] >= self.max_attempts_per_question:
                self.mark_question_skipped(i)
                continue
            return (i, self.questions[i])
        return None

    def get_coverage_ratio(self) -> float:
        """What fraction of questions have been covered."""
        if self.total_questions == 0:
            return 1.0
        return sum(self.questions_covered) / self.total_questions

    def get_remaining_questions(self) -> List[Tuple[int, str]]:
        """Get all remaining uncovered questions."""
        remaining = []
        for i, q in enumerate(self.questions):
            if not self.questions_covered[i]:
                remaining.append((i, q))
        return remaining

    def to_summary(self) -> str:
        """Generate a compact state summary for the LLM prompt."""
        covered = sum(self.questions_covered)
        total = self.total_questions
        lines = [
            f"Assessment progress: {covered}/{total} questions covered ({self.get_coverage_ratio()*100:.0f}%)",
            f"Total exchanges: {self.turn_count}",
        ]
        if self.open_conversation_mode:
            lines.append("Currently in open conversation mode (questionnaire paused)")
        if self.symptoms_explored:
            lines.append(f"Symptom areas explored: {', '.join(self.symptoms_explored[-5:])}")

        remaining = self.get_remaining_questions()
        if remaining:
            next_topics = [q[:60] for _, q in remaining[:3]]
            lines.append(f"Upcoming topics to cover: {'; '.join(next_topics)}")

        return "\n".join(lines)


class AdaptiveTherapistAgent:
    """
    A resistance-aware therapist agent that dynamically adapts its
    therapeutic approach based on detected client resistance.

    Replaces the static MentalHealthAssistant that rigidly follows a
    question script. This agent uses the three-component pipeline:
    1. ResistanceDetector → classify client turn
    2. PolicySelector → choose therapeutic strategy
    3. Response Generator → produce strategy-consistent therapist turn
    """

    def __init__(
        self,
        llm_client,
        model: str,
        questions: List[str],
        rag_engine=None,
        questionnaire_name: str = None,
        policy_mode: PolicyMode = PolicyMode.ADAPTIVE,
        system_prompt: str = None,
    ):
        """
        Args:
            llm_client: LLM client instance (OllamaClient, GroqClient, etc.)
            model: Model name for the LLM
            questions: List of questionnaire questions
            rag_engine: Optional RAG engine for document retrieval
            questionnaire_name: Name of the questionnaire
            policy_mode: Which policy mode to use (adaptive or static baselines)
            system_prompt: Optional custom system prompt
        """
        self.client = llm_client
        self.model = model
        self.questions = questions
        self.rag_engine = rag_engine
        self.questionnaire_name = questionnaire_name

        # Initialize the three components
        self.resistance_detector = ResistanceDetector(
            llm_client=llm_client,
            llm_model=model,
            use_llm_fallback=True,
        )
        self.policy_selector = PolicySelector(mode=policy_mode)
        self.state = ConversationState(questions)

        # Conversation history for LLM context
        self.conversation_history: List[Dict[str, str]] = []

        # Track per-turn metadata for logging
        self.turn_metadata: List[Dict[str, Any]] = []

        # Load or set system prompt
        if system_prompt:
            self.system_prompt = system_prompt
        else:
            self.system_prompt = self._build_system_prompt()

        self.has_introduced = False

        # Seen documents for RAG deduplication
        self.seen_documents = set()

    def _build_system_prompt(self) -> str:
        """Build the base system prompt for the adaptive therapist."""
        return """You are a professional mental health clinician conducting a therapeutic assessment. 
You are skilled in multiple therapeutic approaches including:
- Motivational Interviewing
- Empathic listening and validation
- Autonomy support
- Reflective redirection
- De-escalation techniques

CRITICAL RULES:
1. You are a warm, skilled human therapist — never break character.
2. Ask ONE question or make ONE therapeutic intervention at a time.
3. Always acknowledge the patient's response before moving to a new topic.
4. Adapt your approach based on how the patient is engaging.
5. If the patient is resistant, do NOT just repeat or push through questions.
6. Your goal is both therapeutic engagement AND clinical information gathering.
7. Never diagnose during the conversation — save that for the final assessment.
8. Keep responses concise and natural — avoid lengthy monologues.
9. Never mention questionnaires, forms, assessments explicitly to resistant patients.
10. You are having a CONVERSATION, not administering a test."""

    def generate_introduction(self) -> str:
        """Generate a warm introduction for the assessment."""
        self.has_introduced = True

        # Get questionnaire context if available
        questionnaire_context = ""
        if self.questionnaire_name and self.rag_engine:
            try:
                docs = self.rag_engine.get_context_for_question(
                    f"full text of {self.questionnaire_name}"
                )
                if isinstance(docs, dict) and "content" in docs:
                    questionnaire_context = docs["content"]
                elif isinstance(docs, list) and docs:
                    questionnaire_context = docs[0]
            except Exception as e:
                logger.warning(f"Could not retrieve questionnaire context: {e}")

        intro_prompt = f"""You are a professional mental health clinician beginning an assessment session.

Questionnaire being used: {self.questionnaire_name or 'General mental health assessment'}
Number of topic areas to cover: {len(self.questions)}

{f'Questionnaire content for reference: {questionnaire_context[:1000]}' if questionnaire_context else ''}

Generate a warm, professional introduction that:
1. Introduces yourself naturally as a mental health professional
2. Briefly explains the purpose of this conversation
3. Reassures about confidentiality and creating a safe space
4. Sets expectations that this will be a conversation (not a rigid Q&A)
5. Invites the patient to share what brought them in today

Be warm, natural, and conversational. Keep it to 3-5 sentences.
End by gently opening the conversation — do NOT immediately launch into a clinical question."""

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": intro_prompt},
        ]

        result = self.client.chat(self.model, messages)
        introduction = result.get("response", "")

        self.conversation_history.append({"role": "assistant", "content": introduction})
        return introduction

    def process_patient_response(self, patient_message: str) -> Dict[str, Any]:
        """
        Process a patient's response through the full pipeline:
        1. Detect resistance
        2. Select policy
        3. Generate response

        Args:
            patient_message: The patient's message text

        Returns:
            Dict with keys: 'content' (therapist response), 'resistance',
            'policy', 'metadata', and optionally 'rag_usage'
        """
        self.state.turn_count += 1

        # Add patient message to history
        self.conversation_history.append({"role": "user", "content": patient_message})

        # ========= STEP 1: Resistance Detection =========
        resistance = self.resistance_detector.detect(
            client_message=patient_message,
            conversation_history=self.conversation_history,
        )

        # Update consecutive resistance tracking
        if resistance.resistance_type != ResistanceType.R0_COOPERATIVE:
            self.state.consecutive_resistance_turns += 1
        else:
            self.state.consecutive_resistance_turns = 0

        # ========= STEP 2: Policy Selection =========
        policy_decision = self.policy_selector.select_policy(resistance)

        # Check if we should try to cover a question or go open conversation
        if policy_decision.should_continue_questionnaire:
            self.state.open_conversation_mode = False
        else:
            self.state.open_conversation_mode = True

        # ========= STEP 3: Response Generation =========
        therapist_response, rag_usage = self._generate_response(
            patient_message=patient_message,
            resistance=resistance,
            policy_decision=policy_decision,
        )

        # Update conversation history
        self.conversation_history.append({"role": "assistant", "content": therapist_response})

        # Try to determine if the patient's response covered any questionnaire question
        self._update_question_coverage(
            patient_message=patient_message,
            therapist_message=therapist_response,
            resistance=resistance,
        )

        # Build turn metadata
        turn_meta = {
            "turn": self.state.turn_count,
            "resistance": resistance.to_dict(),
            "policy": policy_decision.to_dict(),
            "state_summary": self.state.to_summary(),
            "questions_covered": sum(self.state.questions_covered),
            "questions_total": self.state.total_questions,
        }
        self.turn_metadata.append(turn_meta)

        result = {
            "content": therapist_response,
            "resistance": resistance,
            "policy": policy_decision,
            "metadata": turn_meta,
        }
        if rag_usage:
            result["rag_usage"] = rag_usage

        return result

    def _generate_response(
        self,
        patient_message: str,
        resistance: ResistanceEstimate,
        policy_decision: PolicyDecision,
    ) -> Tuple[str, Optional[Dict]]:
        """
        Generate a therapist response conditioned on:
        1. The selected strategy bundle (from policy selector)
        2. The conversation state (progress, remaining questions)
        3. Safety constraints
        """
        strategy = policy_decision.strategy_bundle
        state_summary = self.state.to_summary()

        # Determine what clinical question to weave in (if any)
        clinical_direction = ""
        if policy_decision.should_continue_questionnaire:
            next_q = self.state.get_next_uncovered_question()
            if next_q:
                idx, question_text = next_q
                self.state.mark_question_attempted(idx)
                # List already covered topics so the therapist doesn't repeat them
                covered_topics = []
                for i, q in enumerate(self.questions):
                    if self.state.questions_covered[i]:
                        covered_topics.append(q[:60])
                covered_str = ""
                if covered_topics:
                    covered_str = "\n\nTOPICS ALREADY COVERED (do NOT ask about these again):\n" + "\n".join(
                        f"  - {t}" for t in covered_topics)
                clinical_direction = f"""
            NEXT CLINICAL TOPIC TO EXPLORE (weave this into your response naturally):
            "{question_text}"
            Do NOT ask this question verbatim. Adapt it to flow naturally from the conversation.
            If the patient is somewhat resistant, you can soften it or approach it indirectly.
            {covered_str}"""

            else:
                clinical_direction = "All clinical topics have been covered. You may begin wrapping up or ask if there's anything else the patient wants to share."
        else:
            clinical_direction = """
QUESTIONNAIRE IS PAUSED. Do NOT introduce new clinical questions right now.
Focus entirely on the therapeutic strategy (de-escalation, validation, engagement).
You can return to clinical topics once the patient is more engaged."""

        # Build high-severity modifiers
        severity_notes = ""
        if policy_decision.high_severity_modifiers:
            severity_notes = "\nHIGH SEVERITY NOTES:\n" + "\n".join(
                f"  - {m}" for m in policy_decision.high_severity_modifiers
            )

        # Build offer-break note
        break_note = ""
        if policy_decision.should_offer_break:
            break_note = "\nConsider offering the patient a break or a moment of pause."

        # Build the full prompt
        generation_prompt = f"""
{strategy.to_prompt_instructions()}
{severity_notes}
{break_note}

CONVERSATION STATE:
{state_summary}

{clinical_direction}

SAFETY CONSTRAINTS:
- Never make a definitive diagnosis during the conversation
- If the patient mentions self-harm or suicidal ideation, prioritize safety
- Encourage professional help where appropriate
- Be honest about the purpose of the conversation if directly asked

PATIENT'S LAST MESSAGE:
"{patient_message}"

Generate your next therapist response. Be concise (2-4 sentences typically).
Follow the strategy guidelines above. Respond as a skilled human therapist would.
Do NOT include meta-commentary or labels — just the natural response."""

        # Build messages for LLM
        # Include recent conversation history for context
        messages = [{"role": "system", "content": self.system_prompt}]

        # Add recent conversation history (last 10 messages for context)
        recent_history = self.conversation_history[-10:]
        for msg in recent_history[:-1]:  # exclude the last patient message (it's in the prompt)
            messages.append(msg)

        messages.append({"role": "user", "content": generation_prompt})

        # Query RAG if available and we have a clinical direction
        rag_usage = None
        if self.rag_engine and policy_decision.should_continue_questionnaire:
            rag_usage = self._query_rag(patient_message, messages)

        # Generate response
        result = self.client.chat(self.model, messages)
        response = result.get("response", "")

        # Clean up any meta-commentary that leaked through
        response = self._clean_response(response)

        return response, rag_usage

    def _query_rag(self, query: str, messages: List[Dict]) -> Optional[Dict]:
        """Query RAG engine and inject context into messages."""
        try:
            rag_result = self.rag_engine.get_context_for_question(query)

            if isinstance(rag_result, dict) and "content" in rag_result:
                content = rag_result["content"]
                documents = rag_result.get("documents", [])

                # Deduplicate
                new_docs = []
                new_content_parts = []
                for doc in documents:
                    doc_id = doc.get("title", "") + "|" + str(doc.get("highlight", ""))[:50]
                    if doc_id not in self.seen_documents:
                        self.seen_documents.add(doc_id)
                        new_docs.append(doc)

                if new_docs and content:
                    messages.append({
                        "role": "system",
                        "content": f"Additional clinical reference (use if relevant):\n{content[:500]}"
                    })
                    return {
                        "documents": new_docs,
                        "stats": rag_result.get("stats", {}),
                        "count": len(new_docs),
                    }

            elif isinstance(rag_result, list) and rag_result:
                context_str = "\n\n".join(rag_result[:2])
                messages.append({
                    "role": "system",
                    "content": f"Additional clinical reference:\n{context_str[:500]}"
                })
                return {"count": len(rag_result)}

        except Exception as e:
            logger.warning(f"RAG query failed: {e}")

        return None

    def _update_question_coverage(self, patient_message: str, therapist_message: str = None,
                                    resistance: ResistanceEstimate = None):
        """
        Determine if the patient's response (or the exchange as a whole) covers
        any questionnaire questions.

        For resistant patients, we also check the therapist's message — if the
        therapist clearly raised the topic and the patient engaged with it at all
        (even to deny it), the topic was at least attempted meaningfully.
        """
        message_lower = patient_message.lower()
        patient_words = set(re.findall(r'\b\w+\b', message_lower))

        # Also consider the therapist's message if available
        therapist_words = set()
        if therapist_message:
            therapist_words = set(re.findall(r'\b\w+\b', therapist_message.lower()))

        for i, question in enumerate(self.questions):
            if self.state.questions_covered[i]:
                continue
            if not self.state.questions_attempted[i]:
                continue

            question_words = set(re.findall(r'\b\w+\b', question.lower()))
            stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'do', 'does',
                         'did', 'have', 'has', 'had', 'been', 'be', 'being', 'will',
                         'would', 'could', 'should', 'may', 'might', 'can', 'shall',
                         'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                         'or', 'and', 'not', 'no', 'but', 'if', 'than', 'that', 'this',
                         'how', 'much', 'often', 'you', 'your', 'any'}
            question_keywords = question_words - stop_words

            # Method 1: Patient's response has keyword overlap (original check)
            patient_overlap = patient_words & question_keywords
            if len(patient_overlap) >= 3 or (len(question_keywords) <= 3 and len(patient_overlap) >= 2):
                self.state.mark_question_covered(i, patient_message)
                logger.info(f"Question {i} covered via patient keywords (overlap: {patient_overlap})")
                continue

            # Method 2: For resistant patients — if the therapist raised the topic
            # (therapist message has keyword overlap) AND the patient responded at all
            # (even "I'm fine"), count it as covered with the denial noted
            if resistance and resistance.resistance_type != ResistanceType.R0_COOPERATIVE:
                therapist_overlap = therapist_words & question_keywords
                if len(therapist_overlap) >= 2 or (len(question_keywords) <= 3 and len(therapist_overlap) >= 1):
                    # The therapist raised this topic and the patient responded (even with denial)
                    if len(patient_words) >= 2:  # patient said something (not total silence)
                        self.state.mark_question_covered(i, f"[DENIED/RESISTANT] {patient_message}")
                        logger.info(f"Question {i} covered via therapist raising topic + patient denial "
                                   f"(therapist overlap: {therapist_overlap})")
                        continue

    def _clean_response(self, response: str) -> str:
        """Remove meta-commentary, labels, or strategy leakage from the response."""
        # Remove lines that look like meta-labels
        lines = response.split("\n")
        cleaned_lines = []
        for line in lines:
            stripped = line.strip()
            # Skip lines that are just labels
            if re.match(r'^(STRATEGY|APPROACH|NOTE|TECHNIQUE|RESISTANCE|POLICY):', stripped, re.IGNORECASE):
                continue
            if re.match(r'^\[.*\]$', stripped):
                continue
            if stripped.startswith("*") and stripped.endswith("*") and len(stripped) < 50:
                # Likely an italicized meta-note like *pauses thoughtfully*
                # Keep it — it's part of therapeutic communication style
                pass
            cleaned_lines.append(line)

        response = "\n".join(cleaned_lines).strip()

        # Remove any remaining strategy labels in parentheses
        response = re.sub(r'\((?:using|applying|with)\s+(?:MI|motivational interviewing|empathy|validation)\)', '', response, flags=re.IGNORECASE)

        return response.strip()

    """
    PATCH: Replace the generate_diagnosis method in agents/adaptive_therapist.py

    Find this method (around line 509) and replace the ENTIRE method with the version below.
    The key change is that the diagnosis prompt now follows the original framework's format
    with Diagnosis/Reasoning/Recommended Next Steps sections and med/sym/quote tags.
    """

    def generate_diagnosis(self) -> Dict[str, Any]:
        """
        Generate a final diagnosis based on the conversation.
        Uses the same structured format as the original framework.

        Returns:
            Dict with 'content' (diagnosis text) and optionally 'rag_usage'
        """
        # Summarize what was gathered
        covered_responses = []
        for i, q in enumerate(self.questions):
            if self.state.questions_covered[i]:
                resp = self.state.responses.get(i, "Covered indirectly")
                covered_responses.append(f"Q{i + 1}: {q}\nA{i + 1}: {resp[:300]}")

        responses_text = "\n\n".join(
            covered_responses) if covered_responses else "Limited information gathered due to client resistance."

        # First, generate clinical observations (matching original framework)
        observations = self._summarize_observations(responses_text)

        # Build diagnosis prompt matching the ORIGINAL framework format
        diagnosis_prompt = f"""
            Based on the questionnaire responses, please provide a comprehensive mental health assessment.

            Questionnaire responses:
            {responses_text}

            Clinical observations and potential concerns:
            {observations}

            IMPORTANT DIAGNOSTIC CONSIDERATIONS:
            - Consider multiple possible diagnoses that could explain the symptoms
            - Do not default to Somatic Symptom Disorder unless clearly warranted by the symptoms
            - Be open to various diagnostic possibilities including anxiety disorders, mood disorders, trauma-related disorders, etc.
            - Make your diagnosis based solely on the symptoms presented, not on assumptions
            - If symptoms are insufficient for a definitive diagnosis, indicate this is a provisional impression

            Please analyze these responses and observations and provide a professional assessment that MUST follow this EXACT structure:

            1. First paragraph: Write a compassionate summary of what you've heard from the patient, showing empathy for their situation.

            2. After that, include a section with the heading "**Diagnosis:**" (exactly as shown, with the asterisks)
               - On the same line, immediately after the heading, provide the specific diagnosis or clinical impression
               - Do not add extra newlines between the heading and the diagnosis

            3. Next, include a section with the heading "**Reasoning:**" (exactly as shown, with the asterisks)
               - Immediately after this heading, explain your rationale for the diagnosis/impression
               - Do not add extra newlines between the heading and your explanation

            4. Finally, include a section with the heading "**Recommended Next Steps/Treatment Options:**" (exactly as shown, with the asterisks)
               - List specific numbered recommendations (1., 2., 3., etc.)
               - Make each recommendation clear and actionable

            When writing your assessment, use these special tags:
            - Wrap medical terms and conditions in <med>medical term</med> tags
            - Wrap symptoms in <sym>symptom</sym> tags
            - Wrap patient quotes or paraphrases in <quote>patient quote</quote> tags

            EXTREMELY IMPORTANT:
            1. Do NOT include any introductory statements answering the prompt
            2. Do NOT begin with phrases like "Okay, here's a clinical assessment..."
            3. Start DIRECTLY with the compassionate summary paragraph without any preamble
            4. Never include meta-commentary about what you're about to write
            5. Include all four components in the exact order specified
            6. Format section headings consistently with double asterisks
            7. Maintain proper spacing between sections (one blank line)
            8. Do not add extra newlines within sections
            9. Always wrap medical terms, symptoms, and quotes in the specified tags

            Keep your tone professional but warm, showing empathy while maintaining clinical objectivity.
            """

        messages = [
            {"role": "system", "content": "You are a clinical psychologist writing a professional assessment report."},
            {"role": "user", "content": diagnosis_prompt},
        ]

        # RAG for diagnosis
        rag_usage = None
        if self.rag_engine:
            symptoms = " ".join(
                resp[:100] for resp in self.state.responses.values()
            )
            if symptoms:
                rag_usage = self._query_rag(
                    f"mental health diagnosis for patient with symptoms: {symptoms[:300]}",
                    messages,
                )

        result = self.client.chat(self.model, messages)
        diagnosis = result.get("response", "Unable to generate diagnosis.")

        output = {"content": diagnosis}
        if rag_usage:
            output["rag_usage"] = rag_usage

        return output

    def _summarize_observations(self, responses_text: str) -> str:
        """
        Summarize clinical observations from patient responses.
        Matches the original framework's observation generation.

        Args:
            responses_text: Formatted Q&A responses

        Returns:
            str: Clinical observations summary
        """
        summarization_prompt = f"""
            You are a mental health professional reviewing patient responses to a questionnaire.

            Here are the patient's responses:
            {responses_text}

            Based on these responses, please:
            1. Identify the main symptoms and concerns
            2. Note patterns in the patient's responses
            3. List potential areas of clinical significance
            4. Highlight any risk factors or warning signs
            5. Summarize your observations in clinical language

            Format your response as a concise clinical observation summary using professional terminology.
            Focus on extracting the most relevant clinical information while avoiding speculation.
            """

        temp_messages = [
            {"role": "system", "content": "You are a clinical mental health professional conducting an assessment."},
            {"role": "user", "content": summarization_prompt},
        ]

        result = self.client.chat(self.model, temp_messages)
        return result.get("response", "Unable to generate observations.")
    def is_assessment_complete(self) -> bool:
        """Check if the assessment has gathered enough information."""
        # Complete if >80% of questions covered
        if self.state.get_coverage_ratio() >= 0.95:
            return True
        # Or if we've had a very long conversation (diminishing returns)
        if self.state.turn_count >= len(self.questions) * 3:
            return True
        # Or if we've been in open conversation mode for too long with no progress
        if self.state.open_conversation_mode and self.state.consecutive_resistance_turns > 8:
            return True
        return False

    def get_conversation_log(self) -> List[Dict[str, str]]:
        """Get the raw conversation history."""
        return self.conversation_history

    def get_full_metadata(self) -> Dict[str, Any]:
        """Get complete metadata about the conversation for logging."""
        return {
            "turn_count": self.state.turn_count,
            "questions_covered": sum(self.state.questions_covered),
            "questions_total": self.state.total_questions,
            "coverage_ratio": self.state.get_coverage_ratio(),
            "resistance_trend": self.resistance_detector.get_resistance_trend(),
            "policy_summary": self.policy_selector.get_strategy_summary(),
            "turn_by_turn": self.turn_metadata,
        }