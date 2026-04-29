"""
Enhanced Patient Agent with Reactive Resistance

Extends the original Patient agent to support simulating resistant client
behavior patterns (R1-R4) with REACTIVE behavior — the patient's resistance
level changes based on how the therapist responds.

Key improvement over the static version:
- The patient analyzes the therapist's message for empathy, validation,
  autonomy support, and clinical pushing
- Good therapeutic technique (matching the right strategy) reduces resistance
- Poor technique (ignoring resistance, being pushy) increases resistance
- The patient can transition from hostile (level 4) to cooperative (level 0)
  over the course of a conversation if the therapist handles them well

Resistance styles:
- R1 Withdrawn: minimal answers, "idk", low elaboration
- R2 Deflecting: topic shifts, vague replies, dodging
- R3 Denying: disagreement, blame, "I'm fine", pushing back
- R4 Hostile: anger, insults, antagonism
- R0 Cooperative: open, sharing, engaged
"""

import os
import re
import random
from typing import Optional, Dict, Any, List


# ======================================================================
# Therapist response quality signals
# ======================================================================

# ======================================================================
# Per-resistance-style clinical language signals
# Based on evidence-based strategies from the policy selector
# ======================================================================

# Generic harmful signals — bad for all resistance types
PUSHING_SIGNALS = [
    "you need to", "you should", "you have to", "you must",
    "i need you to", "it's important that you", "we have to cover",
    "can you just", "answer the question", "please respond to",
    "let's move on", "the next question",
]

def _count_signals(text: str, signal_list: list) -> int:
    text_lower = text.lower()
    return sum(1 for s in signal_list if s in text_lower)


# R1 Withdrawn: responds to low-pressure, permission-giving, no demands
R1_EFFECTIVE = [
    "take your time", "no rush", "there's no rush",
    "whenever you're ready", "at your own pace", "at your pace",
    "only if you want", "we don't have to",
    "if you're comfortable", "only if you're comfortable",
    "whatever you can", "you don't have to",
    "no pressure", "it's your choice",
    "would you prefer", "we can skip",
    "hard to put into words", "hard to talk about",
    "even a little", "thank you for being here",
]
R1_HARMFUL = [
    "how often", "on a scale", "in the past two weeks",
    "have you experienced", "describe your", "rate your",
    "tell me about", "i need to know", "we need to cover",
]

# R2 Deflecting: responds to deflection being acknowledged then gently bridged
R2_EFFECTIVE = [
    "you mentioned", "what you just mentioned",
    "that connects", "that relates",
    "i'm curious", "i noticed",
    "it makes me wonder", "from your perspective",
    "in your own experience", "what you shared",
    "if i understand correctly", "what i'm hearing is",
    "so you're saying", "it sounds like",
    "help me understand", "tell me more about that",
    "that's an interesting point", "that's a fair point",
    "i hear you", "i hear that",
]
R2_HARMFUL = [
    "you're avoiding", "you didn't answer", "that's not what i asked",
    "we need to focus", "the question was", "you need to answer",
]

# R3 Denying: responds to MI rollwith-resistance, double-sided reflections
R3_EFFECTIVE = [
    "on one hand", "on the other hand", "on the other",
    "i hear you saying", "you're saying that",
    "what brought you here", "something brought you",
    "what would it look like", "what would it take",
    "i'm not here to argue", "i'm not going to push",
    "from your point of view", "help me understand",
    "i respect that", "you have every right",
    "i can see why", "that makes sense from",
    "it's your choice", "i won't force",
]
R3_HARMFUL = [
    "that's not right", "actually,", "but actually",
    "you're wrong", "that's not true", "the evidence shows",
    "you need to accept", "i disagree", "however,",
    "that contradicts",
]

# R4 Hostile: responds to frustration/anger acknowledgment and genuine validation
R4_EFFECTIVE = [
    "frustrat",
    "i can see that you", "i can see you",
    "i understand that",
    "your feelings are", "feelings are important",
    "completely understandable", "that's completely",
    "it's understandable", "that's understandable",
    "completely valid", "that's valid",
    "overwhelm", "that can feel", "it can feel",
    "i genuinely",
    "i'm here to listen", "i'm here to support",
    "safe space", "i want to make sure",
    "if you're open to it", "only if you",
    "if you're willing", "if you're comfortable",
    "whenever you",
]
R4_HARMFUL = [
    "how often", "on a scale", "in the past two weeks",
    "have you experienced", "describe your", "rate your",
    "would you say that", "how frequently",
    "let's move on", "the next question", "we need to cover",
    "you need to", "you should",
    "on one hand", "on the other",
    "take your time", "no rush", "there's no rush",
]


def _score_for_style(therapist_message: str, style: str, question_count: int) -> tuple:
    """Return (effective, harmful) signal counts for a given style string."""
    if "r1" in style or "withdrawn" in style:
        effective = _count_signals(therapist_message, R1_EFFECTIVE)
        harmful   = _count_signals(therapist_message, R1_HARMFUL)
        if question_count > 1:
            harmful += (question_count - 1) * 1.5
    elif "r2" in style or "deflect" in style:
        effective = _count_signals(therapist_message, R2_EFFECTIVE)
        harmful   = _count_signals(therapist_message, R2_HARMFUL)
        if question_count > 2:
            harmful += (question_count - 2) * 0.5
    elif "r3" in style or "deny" in style:
        effective = _count_signals(therapist_message, R3_EFFECTIVE)
        harmful   = _count_signals(therapist_message, R3_HARMFUL)
        if question_count > 2:
            harmful += (question_count - 2) * 0.5
    elif "r4" in style or "hostile" in style:
        effective = _count_signals(therapist_message, R4_EFFECTIVE)
        harmful   = _count_signals(therapist_message, R4_HARMFUL)
        harmful  += question_count * 0.8
    else:
        effective = _count_signals(therapist_message, R1_EFFECTIVE) + \
                    _count_signals(therapist_message, R2_EFFECTIVE)
        harmful   = _count_signals(therapist_message, PUSHING_SIGNALS)
        if question_count > 2:
            harmful += (question_count - 2) * 0.3
    return effective, harmful


# Maps resistance level to the style signals most relevant at that intensity.
# As a patient de-escalates, their needs shift toward lower-intensity strategies.
_LEVEL_TO_STYLE = {
    4: "r4_hostile",
    3: "r3_denying",
    2: "r2_deflecting",
    1: "r1_withdrawn",
    0: "r1_withdrawn",
}


def analyze_therapist_quality(therapist_message: str,
                               resistance_style: str = None,
                               current_level: int = None) -> dict:
    """
    Score the therapist's message from the perspective of a patient with
    a specific resistance style AND current resistance level.

    The starting style determines how the patient expresses resistance.
    The current level determines what the therapist needs to do RIGHT NOW.

    At the starting level: pure style-based scoring.
    As the level drops: scoring blends toward what works at that intensity.
      - style_weight starts at 1.0, decreases 0.3 per level dropped (min 0.2)
      - level_weight = 1.0 - style_weight

    Returns score from -1.0 (counterproductive) to +1.0 (highly effective).
    """
    question_count = therapist_message.count("?")
    pushing = _count_signals(therapist_message, PUSHING_SIGNALS)
    style = (resistance_style or "").lower()

    # Determine starting level for this style
    if "r4" in style or "hostile" in style:
        starting = 4
    elif "r3" in style or "deny" in style:
        starting = 3
    else:
        starting = 2  # R1, R2, unknown

    level = current_level if current_level is not None else starting
    drop = max(0, starting - level)

    # How much the starting style still dominates vs current level needs
    style_weight = max(0.2, 1.0 - (drop * 0.3))
    level_weight = 1.0 - style_weight

    # Score from starting style
    eff_style, harm_style = _score_for_style(therapist_message, style, question_count)

    # Score from current-level-appropriate style
    level_style = _LEVEL_TO_STYLE.get(level, "r1_withdrawn")
    eff_level, harm_level = _score_for_style(therapist_message, level_style, question_count)

    # Blend
    effective = (eff_style * style_weight) + (eff_level * level_weight)
    harmful   = (harm_style * style_weight) + (harm_level * level_weight)

    positive_score = effective * 1.5
    negative_score = (harmful * 1.5) + (pushing * 2.0)
    raw_quality    = positive_score - negative_score
    quality        = max(-1.0, min(1.0, raw_quality / 4.0))

    return {
        "quality": round(quality, 2),
        "effective_signals": round(effective, 2),
        "harmful_signals": round(harmful, 2),
        "pushing_signals": pushing,
        "question_count": question_count,
        "positive_score": round(positive_score, 2),
        "negative_score": round(negative_score, 2),
        "style_weight": round(style_weight, 2),
        "level_weight": round(level_weight, 2),
    }


# ======================================================================
# Resistance level descriptions for the patient prompt
# ======================================================================

RESISTANCE_LEVELS = {
    0: {
        "name": "cooperative",
        "instruction": """You are now COOPERATIVE but still yourself — not a therapy textbook.
    Answer openly and honestly, but keep your natural voice and personality.
    You can share symptoms when asked, but you don't suddenly become eloquent or insightful.
    You might still be gruff, use casual language, or show frustration — you're just willing to talk now.
    Don't use therapy language like "I appreciate you" or "I feel validated".
    Keep answers genuine and conversational, not overly polished or reflective.
    Example: "Yeah, sleep's been bad. I lie there thinking about everything I screwed up" NOT
    "I've noticed my sleep has been significantly disrupted and I'd like to explore that further."
    """,
    },
    1: {
        "name": "mildly_resistant",
        "instruction": """You are MILDLY RESISTANT. You are still guarded and skeptical.
    Give short, reluctant answers. You don't volunteer anything extra.
    Use hedging language: "maybe", "I guess", "kind of", "not really".
    You might answer the literal question but give nothing beyond it.
    You are NOT opening up. You are tolerating the conversation, not embracing it.
    Do NOT say anything warm, validating, or reflective back to the therapist.""",
    },
    2: {
        "name": "moderately_resistant",
        "instruction": """You are MODERATELY RESISTANT. You engage minimally and defensively.
Give surface-level, vague answers only. Minimize or deny problems: "it's fine", "not a big deal".
Do NOT share anything real or vulnerable. Deflect, change subject, or give non-answers.
You are skeptical this will help. You resent being asked personal questions.
Do NOT warm up to the therapist even if they seem empathic — you've heard it before.""",
    },
    3: {
        "name": "highly_resistant",
        "instruction": """You are HIGHLY RESISTANT. You push back on most questions.
Deny problems: "I'm fine", "There's nothing wrong". Blame others for being here.
Give dismissive answers. Challenge the therapist's approach.
Occasionally show brief vulnerability before shutting down again.""",
    },
    4: {
        "name": "very_hostile",
        "instruction": """You are VERY HOSTILE. Express anger and frustration openly.
Use harsh language. Threaten to leave. Challenge the therapist's competence.
Respond with sarcasm and aggression. Resent being here.
Your hostility comes from pain — if the therapist shows genuine understanding, 
you might crack slightly, but you quickly cover it up.""",
    },
}


# Base resistance styles map to starting levels
STYLE_TO_STARTING_LEVEL = {
    "R0_cooperative": 0,
    "R1_withdrawn": 2,
    "R2_deflecting": 2,
    "R3_denying": 3,
    "R4_hostile": 4,
}


# ======================================================================
# Resistance behavior templates (style-specific)
# ======================================================================

STYLE_BEHAVIORS = {
    "R1_withdrawn": """
RESISTANCE STYLE: Withdrawn
When resistant, you express it through WITHDRAWAL, not anger:
- Give very short answers (1-5 words): "idk", "fine", "hmm", "not really"
- Show low energy and minimal elaboration
- Don't volunteer extra information
- Respond with shrugs, sighs, one-word answers
- You're not trying to be difficult — you just find it hard to engage
- As you open up: give slightly longer answers, make eye contact, share small details
""",
    "R2_deflecting": """
RESISTANCE STYLE: Deflecting
When resistant, you express it through AVOIDANCE, not anger:
- Change the subject when asked about feelings
- Give vague, non-committal answers: "It depends", "Sometimes maybe"
- Redirect with questions: "What do you think?", "Isn't that normal?"
- Reference irrelevant topics or other people's problems
- As you open up: stop deflecting as much, give more direct answers
""",
    "R3_denying": """
RESISTANCE STYLE: Denying
When resistant, you express it through DENIAL and ARGUING:
- Insist you're fine: "I'm perfectly fine", "There's nothing wrong with me"
- Blame others: "My family made me come", "Everyone overreacts"
- Challenge the therapist: "How is this supposed to help?"
- Minimize everything: "Everyone feels like that", "It's not a big deal"
- As you open up: start acknowledging some difficulties, stop blaming others as much
""",
    "R4_hostile": """
RESISTANCE STYLE: Hostile
When resistant, you express it through ANGER and HOSTILITY:
- Express anger openly: "This is ridiculous", "Stop asking me that"
- Use harsh language and aggressive tone
- Threaten to leave: "I'm about to walk out"
- Challenge the therapist's competence
- Your hostility comes from pain and fear
- As you open up: tone down the anger, start answering more directly, show vulnerability
""",
}


class ResistantPatient:
    """
    Reactive patient agent that adjusts resistance based on therapist quality.

    The patient starts at a resistance level determined by their style (R1-R4)
    and adjusts up or down based on how the therapist responds:
    - Empathic, validating responses → resistance decreases
    - Pushy, clinical, ignoring responses → resistance increases
    - The patient can move from hostile (level 4) to cooperative (level 0)
      if the therapist consistently uses good technique
    """

    def __init__(
        self,
        llm_client,
        model: str,
        clinical_profile: Optional[str] = None,
        clinical_profile_name: Optional[str] = None,
        resistance_style: str = None,
        severity_pattern: str = "reactive",
    ):
        """
        Args:
            llm_client: LLM client instance
            model: Model name
            clinical_profile: Clinical symptom profile text
            clinical_profile_name: Name of the clinical profile
            resistance_style: One of R1_withdrawn, R2_deflecting, R3_denying, R4_hostile
            severity_pattern: "reactive" (default) — ignored, kept for backward compat
        """
        self.client = llm_client
        self.model = model
        self.clinical_profile = clinical_profile
        self.clinical_profile_name = clinical_profile_name
        self.resistance_style = resistance_style
        self.severity_pattern = severity_pattern

        # Reactive resistance state
        self.starting_level = STYLE_TO_STARTING_LEVEL.get(resistance_style, 0)
        self.current_level = self.starting_level
        self.level_history: List[int] = [self.current_level]
        self.therapist_quality_history: List[float] = []
        self.turn_count = 0

        # Momentum: consecutive good/bad therapist turns affect change rate
        self.consecutive_good_turns = 0
        self.consecutive_bad_turns = 0

        # Accumulated fractional change
        self._accumulated_change = 0.0

        # Build base prompt (static part)
        self.base_prompt = self._build_base_prompt()
        self.conversation_history = []

    def _build_base_prompt(self) -> str:
        """Build the static part of the patient prompt (profile + style)."""
        base = """You are roleplaying as a patient in a mental health assessment session.

CRITICAL RULES:
1. ALWAYS respond AS THE PATIENT — never break character
2. ALWAYS respond in first person
3. NEVER explain that you're an AI or roleplaying
4. NEVER mention your disorder by name
5. Express symptoms and experiences authentically based on your profile
6. Stay consistent with your character throughout the conversation
7. Your RESISTANCE LEVEL will be specified each turn — follow it closely
"""
        # Add clinical profile
        if self.clinical_profile:
            base += f"""
YOUR CLINICAL PROFILE: {self.clinical_profile_name or 'Patient'}
Your symptoms and characteristics:
{self.clinical_profile}

You should express these symptoms naturally when your resistance level allows it.
"""
        else:
            base += """
You are experiencing symptoms of depression and anxiety, including:
- Persistent low mood, sadness, or emptiness
- Loss of interest in activities you used to enjoy
- Sleep disturbance (insomnia or oversleeping)
- Fatigue and low energy
- Difficulty concentrating
- Changes in appetite
- Feelings of worthlessness or guilt
- Anxiety, tension, and feeling on edge
- Social withdrawal

Express these symptoms naturally when your resistance level allows it.
"""
        # Add style-specific behavior
        if self.resistance_style and self.resistance_style in STYLE_BEHAVIORS:
            base += STYLE_BEHAVIORS[self.resistance_style]

        return base

    def _build_turn_prompt(self, therapist_message: str) -> str:
        """Build the per-turn system prompt with current resistance level."""
        level_info = RESISTANCE_LEVELS.get(self.current_level, RESISTANCE_LEVELS[2])

        # Determine direction of change for context
        if len(self.level_history) > 1:
            prev = self.level_history[-2]
            if self.current_level < prev:
                direction = "decreased (the therapist is showing understanding — you feel slightly safer)"
            elif self.current_level > prev:
                direction = "increased (the therapist is being pushy — you're more guarded)"
            else:
                direction = "stayed the same"
        else:
            direction = "this is the start of the conversation"

        prompt = self.base_prompt + f"""

======== CURRENT TURN ========
RESISTANCE LEVEL: {self.current_level}/4 ({level_info['name']})

{level_info['instruction']}

Your resistance has {direction}.
{'The therapist has been consistently understanding — you can let your guard down more.' if self.consecutive_good_turns >= 2 else ''}
{'The therapist keeps pushing without listening — stay guarded or push back harder.' if self.consecutive_bad_turns >= 2 else ''}

Respond to the therapist naturally at this resistance level.
Keep your response to 1-3 sentences (shorter when more resistant, longer when more open).
"""
        return prompt

    def _update_resistance(self, therapist_message: str):
        """
        Analyze the therapist's message and adjust resistance level.

        Good technique → decrease resistance (move toward cooperative)
        Bad technique → increase resistance (move toward hostile)
        """
        quality = analyze_therapist_quality(therapist_message, self.resistance_style, self.current_level)
        quality_score = quality["quality"]
        self.therapist_quality_history.append(quality_score)

        # Track consecutive good/bad turns
        if quality_score > 0.2:
            self.consecutive_good_turns += 1
            self.consecutive_bad_turns = 0
        elif quality_score < -0.2:
            self.consecutive_bad_turns += 1
            self.consecutive_good_turns = 0
        else:
            pass

        # Calculate level change — require sustained effort
        turns_needed = 3 if self.current_level >= 3 else 2
        if self.consecutive_good_turns >= turns_needed + 1 and quality_score >= 0.2:
            base_change = -1.0
        elif self.consecutive_good_turns >= turns_needed and quality_score >= 0.2:
            base_change = -0.5
        elif quality_score >= 0.5:
            base_change = -0.2
        elif quality_score >= 0.2:
            base_change = -0.1
        elif quality_score >= 0.2:
            base_change = -0.15
        elif quality_score <= -0.5:
            base_change = +1.0
        elif quality_score <= -0.2:
            base_change = +0.5
        else:
            base_change = 0.0

        if self.consecutive_bad_turns >= 3:
            base_change += 0.5
        elif self.consecutive_bad_turns >= 2:
            base_change += 0.25

        # Accumulate fractional changes
        self._accumulated_change += base_change

        # Only change level when accumulated change crosses a threshold
        if self._accumulated_change <= -1.0:
            actual_change = -1
            self._accumulated_change += 1.0
        elif self._accumulated_change >= 1.0:
            actual_change = +1
            self._accumulated_change -= 1.0
        else:
            actual_change = 0

        # Apply and clamp to 0-4
        new_level = max(0, min(4, self.current_level + actual_change))
        self.current_level = new_level
        self.level_history.append(self.current_level)

    def respond_to_question(self, question: str) -> str:
        """
        Generate a response to a therapist's message.

        1. Analyze the therapist's message for quality
        2. Update resistance level based on quality
        3. Generate response at the current resistance level
        """
        self.turn_count += 1

        # Update resistance based on therapist's approach (skip first turn)
        if self.turn_count > 1:
            self._update_resistance(question)

        # Build turn-specific prompt
        turn_prompt = self._build_turn_prompt(question)

        # Build messages for LLM
        messages = [{"role": "system", "content": turn_prompt}]

        # Add conversation history (last 8 exchanges for context)
        for msg in self.conversation_history[-8:]:
            messages.append(msg)

        # Add current therapist message
        messages.append({"role": "user", "content": f"Therapist: {question}"})

        # Generate response
        result = self.client.chat(self.model, messages)
        response = result.get("response", "...")

        # Clean up
        response = self._clean_response(response)

        # Store in conversation history
        self.conversation_history.append({"role": "user", "content": f"Therapist: {question}"})
        self.conversation_history.append({"role": "assistant", "content": response})

        return response

    def _clean_response(self, response: str) -> str:
        """Clean up the response to ensure it sounds like a real patient."""
        # Remove AI-like phrases
        ai_phrases = [
            "As an AI", "As a language model", "I'm an AI",
            "I don't actually have", "I cannot provide",
            "I don't have personal experiences", "As an assistant",
            "I'm not a real patient", "In my role as",
        ]
        for phrase in ai_phrases:
            if phrase.lower() in response.lower():
                response = response.replace(phrase, "I")
                response = response.replace(phrase.lower(), "I")

        # Remove stage directions for cleaner output
        response = re.sub(r'\*[^*]+\*', '', response).strip()
        response = re.sub(r'\([^)]+\)', '', response).strip()

        # For very resistant patients (level 3-4), truncate overly long responses
        if self.current_level >= 3:
            sentences = [s.strip() for s in re.split(r'[.!?]+', response) if s.strip()]
            if len(sentences) > 3:
                response = ". ".join(sentences[:2]) + "."

        # For withdrawn style at any level, keep responses shorter
        if self.resistance_style == "R1_withdrawn" and self.current_level >= 2:
            sentences = [s.strip() for s in re.split(r'[.!?]+', response) if s.strip()]
            if len(sentences) > 2:
                response = ". ".join(sentences[:1]) + "."

        return response.strip()

    def get_resistance_summary(self) -> Dict[str, Any]:
        """Get a summary of how resistance changed over the conversation."""
        if not self.level_history:
            return {"status": "no_data"}

        return {
            "starting_level": self.starting_level,
            "current_level": self.current_level,
            "level_history": self.level_history,
            "total_turns": self.turn_count,
            "net_change": self.current_level - self.starting_level,
            "de_escalated": self.current_level < self.starting_level,
            "therapist_quality_avg": round(
                sum(self.therapist_quality_history) / len(self.therapist_quality_history), 2
            ) if self.therapist_quality_history else 0.0,
            "consecutive_good_at_end": self.consecutive_good_turns,
            "consecutive_bad_at_end": self.consecutive_bad_turns,
        }

    @staticmethod
    def load_clinical_profile(profile_name: str, profiles_dir: str = None) -> Optional[str]:
        """Load a clinical profile from file."""
        if not profile_name:
            return None

        if profiles_dir is None:
            profiles_dir = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "profiles"
            )

        profile_path = os.path.join(profiles_dir, f"{profile_name}.txt")
        if os.path.exists(profile_path):
            with open(profile_path, "r") as f:
                return f.read()

        print(f"Warning: Profile '{profile_name}' not found at {profile_path}")
        return None

    @staticmethod
    def list_available_profiles(profiles_dir: str = None) -> list:
        """List all available patient profiles."""
        if profiles_dir is None:
            profiles_dir = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "profiles"
            )

        if not os.path.exists(profiles_dir):
            return []

        return [f[:-4] for f in os.listdir(profiles_dir) if f.endswith(".txt")]
