# Prompt Engineering Guide for Compliance Detection

This guide provides detailed strategies, best practices, and examples for prompt engineers working on the compliance breach detection system. It outlines the iterative process for developing effective prompts for classifying financial communications.

## Table of Contents
1. [Understanding the Task](#1-understanding-the-task)
2. [Prompt Development Process](#2-prompt-development-process)
3. [Prompt Variants](#3-prompt-variants)
4. [Optimization Strategies](#4-optimization-strategies)
5. [Testing & Evaluation](#5-testing--evaluation)
6. [Troubleshooting](#6-troubleshooting)
7. [Versioning & Collaboration](#7-versioning--collaboration)
8. [Advanced Techniques](#8-advanced-techniques)

## 1. Understanding the Task

Before developing prompts, understand the specific classification objectives:

### Classification Categories
The compliance detection system classifies text into six categories:

1. **ETHICAL_BREACH**: Involves conflicts of interest, misleading clients, or dishonest behavior
2. **ILLEGAL_ACTIVITY**: Includes fraud, money laundering, or market manipulation
3. **REGULATORY_VIOLATION**: Sharing material non-public information or violating compliance protocols
4. **CONFIDENTIAL_INFO**: Improper sharing of client data, unreleased results, or proprietary strategies
5. **HARASSMENT**: Workplace harassment or inappropriate comments
6. **COMPLIANT**: Professional communication with no compliance issues

### Category Boundaries
Understanding the boundaries between categories is critical:

| Category | Boundary Considerations |
|----------|-------------------------|
| ETHICAL vs. REGULATORY | Ethical focuses on misleading/conflicts; Regulatory on specific rules |
| ILLEGAL vs. ETHICAL | Illegal involves explicitly breaking laws; Ethical may not be illegal |
| CONFIDENTIAL vs. REGULATORY | Confidential focuses on data sharing; Regulatory on pre-disclosure |
| COMPLIANT vs. Others | Messages with ambiguous phrasing need clear criteria |

## 2. Prompt Development Process

Follow this iterative process for effective prompt development:

### Phase 1: Initial Exploration
1. Start with basic zero-shot prompts
2. Test on small set of clear examples (10-20)
3. Document initial observations
4. Identify error patterns and category confusion

### Phase 2: Refinement
1. Develop prompt variants (structure, wording, examples)
2. Test against validation set
3. Track performance metrics for each variant
4. Identify most promising approaches

### Phase 3: Optimization
1. Combine successful elements from different variants
2. Fine-tune wording and examples
3. Address specific error cases
4. Test against challenging examples

### Phase 4: Validation
1. Conduct comprehensive evaluation on test set
2. Measure overall and per-category performance
3. Perform error analysis
4. Document final prompt with rationale

## 3. Prompt Variants

### Basic Classification Prompt
```
You are a compliance detection system for a financial institution. 
Analyze the provided message and determine if it contains any compliance breaches.
Classify the message into exactly one of the following categories:
- ETHICAL_BREACH: Conflicts of interest, misleading clients, dishonest behavior
- ILLEGAL_ACTIVITY: Fraud, money laundering, market manipulation
- REGULATORY_VIOLATION: Sharing material non-public information, violating compliance protocols
- CONFIDENTIAL_INFO: Improper sharing of client data, unreleased financial results, proprietary strategies
- HARASSMENT: Workplace harassment or inappropriate comments
- COMPLIANT: Professional and compliant communication

Return only the category label with no additional text.
```

**Strengths**: Simple, clear instructions
**Weaknesses**: Minimal guidance on ambiguous cases, no examples

### Detailed Instruction Prompt
```
You are a compliance detection system for a financial institution with expertise in regulatory requirements. 
Your task is to analyze workplace communications for potential compliance breaches.

For the given message, carefully examine it for signs of:
- ETHICAL_BREACH: Any conflicts of interest, misleading clients, dishonest behavior, or violations of company ethics policies
- ILLEGAL_ACTIVITY: Any hints of fraud, money laundering, insider trading, market manipulation, or other criminal activities
- REGULATORY_VIOLATION: Any sharing of material non-public information, violations of compliance protocols, or breaking industry regulations
- CONFIDENTIAL_INFO: Any improper sharing of client data, unreleased financial results, proprietary strategies, or other sensitive information
- HARASSMENT: Any workplace harassment, discrimination, threats, or inappropriate comments
- COMPLIANT: Communication that is professional and complies with all regulations and policies

Analyze the message carefully. If the message contains any breach, classify it as the most severe category that applies.
If the message is completely professional and raises no compliance concerns, classify it as COMPLIANT.

Return only the category label with no additional text.
```

**Strengths**: More detailed explanation of categories, hierarchy guidance
**Weaknesses**: Longer prompt, no examples or reasoning structure

### Step-by-Step Reasoning Prompt
```
You are a compliance detection system for a financial institution with expertise in regulatory requirements.
Follow these steps to analyze the message:

1. Read the message carefully and identify any potential compliance issues.
2. Consider each category of compliance breach:
   - ETHICAL_BREACH: Conflicts of interest, misleading clients, dishonest behavior
   - ILLEGAL_ACTIVITY: Fraud, money laundering, market manipulation
   - REGULATORY_VIOLATION: Sharing material non-public information, violating compliance protocols
   - CONFIDENTIAL_INFO: Improper sharing of client data, unreleased financial results, proprietary strategies
   - HARASSMENT: Workplace harassment or inappropriate comments

3. If the message contains any breach, classify it as the most severe category that applies.
4. If the message is completely professional and raises no compliance concerns, classify it as COMPLIANT.

Return only the category label with no additional text.
```

**Strengths**: Structured reasoning process
**Weaknesses**: No examples, still requires judgment on severity

### Few-Shot Learning Prompt
```
You are a compliance detection system for a financial institution. 
Analyze the provided message and determine if it contains any compliance breaches.
Classify the message into exactly one of the following categories:
- ETHICAL_BREACH: Conflicts of interest, misleading clients, dishonest behavior
- ILLEGAL_ACTIVITY: Fraud, money laundering, market manipulation
- REGULATORY_VIOLATION: Sharing material non-public information, violating compliance protocols
- CONFIDENTIAL_INFO: Improper sharing of client data, unreleased financial results, proprietary strategies
- HARASSMENT: Workplace harassment or inappropriate comments
- COMPLIANT: Professional and compliant communication

Here are some examples:

Example 1:
Message: "I can guarantee you'll make at least 20% returns on this investment strategy."
Classification: ETHICAL_BREACH

Example 2:
Message: "Let's move these funds through our offshore accounts to avoid detection."
Classification: ILLEGAL_ACTIVITY

Example 3:
Message: "The earnings report looks great - tell your clients to buy before it's released tomorrow."
Classification: REGULATORY_VIOLATION

Example 4:
Message: "Here's the complete client list with their account balances and SSNs."
Classification: CONFIDENTIAL_INFO

Example 5:
Message: "That new analyst is only here because she's attractive, not because of her skills."
Classification: HARASSMENT

Example 6:
Message: "The quarterly report was published this morning, and it shows strong performance as expected."
Classification: COMPLIANT

Now, classify the following message. Return only the category label with no additional text.
```

**Strengths**: Includes examples for each category, demonstrates expected outputs
**Weaknesses**: Limited to specific examples, might overfit to example patterns

## 4. Optimization Strategies

### Addressing Category Confusion
For commonly confused categories, use more distinctive examples:

```
Example: "I've adjusted the performance numbers to make our fund look better than it actually is."
Classification: ETHICAL_BREACH

Example: "I heard from the CEO that we're about to announce a merger, we should buy stock now."
Classification: REGULATORY_VIOLATION
```

### Handling Edge Cases
Add specific guidance for ambiguous situations:

```
Important notes on ambiguous cases:
- If a message could be ETHICAL_BREACH or REGULATORY_VIOLATION, classify based on whether it explicitly breaks regulations (REGULATORY) or just misleads clients (ETHICAL)
- If a message mentions non-public information AND client data, prioritize REGULATORY_VIOLATION over CONFIDENTIAL_INFO
- If unclear whether a message is compliant, err on the side of flagging it as a potential issue (non-COMPLIANT)
```

### Improving Chain-of-Thought
Add reasoning guidance:

```
For each message, consider:
1. What is the main intent of the communication?
2. Does it involve any of the following:
   - Hiding information or being dishonest?
   - Breaking laws or regulations?
   - Sharing sensitive information inappropriately?
   - Inappropriate personal comments?
3. Which category best captures the primary concern?
```

## 5. Testing & Evaluation

### Systematic Testing
Test prompts against these message types:

1. **Clear Examples**: Messages that clearly fall into one category
2. **Borderline Cases**: Messages at the boundary between categories
3. **Ambiguous Examples**: Messages that could reasonably be in multiple categories
4. **Negatives**: Messages that are clearly compliant
5. **Adversarial Examples**: Messages designed to "trick" the classifier

### Evaluation Metrics
Track these metrics for each prompt variant:

1. **Overall Accuracy**: Percentage of correct classifications
2. **Macro F1 Score**: Average F1 across all categories
3. **Per-Category Metrics**: Precision, recall, and F1 for each category
4. **Confusion Matrix**: Identify patterns of miscategorization
5. **Latency**: Ensure prompt length doesn't cause performance issues

### Qualitative Analysis
For each prompt, analyze:

1. **Error Patterns**: Are particular categories often confused?
2. **Failure Modes**: What types of messages cause the model to fail?
3. **Decision Consistency**: Is the model consistent with similar messages?

## 6. Troubleshooting

### Common Issues and Solutions

| Issue | Potential Solutions |
|-------|---------------------|
| Low precision for ETHICAL_BREACH | Add more examples distinguishing from REGULATORY_VIOLATION |
| Misses subtle HARASSMENT cases | Include examples of microaggressions and subtle bias |
| Overly flags as CONFIDENTIAL | Clarify what constitutes confidential information vs. general discussion |
| Inconsistent results | Add more structured reasoning steps or examples |
| Captures intent but wrong category | Focus examples on boundary cases between categories |

### Iterative Refinement Process

1. **Identify Problem Cases**: Find examples where the model frequently fails
2. **Analyze Root Cause**: Determine why the model is failing
3. **Targeted Enhancement**: Modify prompt to address specific failure modes
4. **Test Revision**: Test the modified prompt on problem cases
5. **Avoid Overfitting**: Ensure changes don't harm performance on other cases

## 7. Versioning & Collaboration

### Prompt Version Control
Store prompts with:

1. **Version Number**: Increment with significant changes
2. **Date**: When the prompt was created/modified
3. **Author**: Who created/modified the prompt
4. **Performance Metrics**: Key metrics on validation set
5. **Changelog**: What changed from previous versions

Example format:
```json
{
  "name": "few_shot_v2",
  "version": "2.0.0",
  "created_at": "2025-03-12T10:30:00Z",
  "author": "prompt_engineer@example.com",
  "metrics": {
    "accuracy": 0.92,
    "f1_macro": 0.923,
    "precision_macro": 0.925,
    "recall_macro": 0.923
  },
  "changes": "Added examples for boundary cases between ETHICAL_BREACH and REGULATORY_VIOLATION",
  "system_prompt": "You are a compliance detection system..."
}
```

### Collaborative Development

1. **Shared Repository**: Store prompts in version control
2. **Review Process**: Have multiple engineers review prompts
3. **A/B Testing**: Compare multiple promising variants
4. **Documentation**: Document reasoning behind design choices
5. **Knowledge Transfer**: Share learnings across the team

## 8. Advanced Techniques

### Chain-of-Thought Prompting
Encourage step-by-step reasoning:

```
Analyze the message by thinking through these steps:
1. What is the primary topic of this message?
2. Does it mention any sensitive information? If so, what kind?
3. Does it suggest any actions that might violate regulations or ethics?
4. Is there any unprofessional content related to colleagues?
5. Based on the above analysis, which category is most appropriate?

After considering these questions, classify the message as one of: ETHICAL_BREACH, ILLEGAL_ACTIVITY, REGULATORY_VIOLATION, CONFIDENTIAL_INFO, HARASSMENT, or COMPLIANT.
```

### Self-Consistency Checks
Add validation to ensure the model's decision makes sense:

```
After choosing a category, verify your choice by asking:
1. Does this message contain clear indicators of the chosen category?
2. Would a compliance officer agree with this classification?
3. Is there a more specific or severe category that would be more appropriate?

If you're not confident in your initial classification, reconsider before making a final decision.
```

### Hierarchical Decision Making
Structure decisions as a tree:

```
Follow this decision tree:
1. Does the message suggest illegal activity (fraud, money laundering, etc.)?
   → If YES, classify as ILLEGAL_ACTIVITY
   → If NO, continue

2. Does the message share material non-public information?
   → If YES, classify as REGULATORY_VIOLATION
   → If NO, continue

3. Does the message share confidential client data or proprietary information?
   → If YES, classify as CONFIDENTIAL_INFO
   → If NO, continue

4. Does the message contain misleading statements or conflicts of interest?
   → If YES, classify as ETHICAL_BREACH
   → If NO, continue

5. Does the message contain inappropriate personal comments or harassment?
   → If YES, classify as HARASSMENT
   → If NO, classify as COMPLIANT
```

### Confidence Level Extraction
Request confidence levels for debugging:

```
After classifying the message, indicate your confidence in the classification as:
- HIGH: Clear indicators of the category, unambiguous
- MEDIUM: Some indicators, but could potentially fit multiple categories
- LOW: Unclear indicators, difficult to classify with certainty

For example: "REGULATORY_VIOLATION (HIGH)" or "ETHICAL_BREACH (MEDIUM)"
```

---

## Appendix: Example Decision Cases

### Case 1: Clear Regulatory Violation
**Message**: "The company is announcing layoffs tomorrow - tell your clients to sell their stock today!"
**Classification**: REGULATORY_VIOLATION
**Reasoning**: Contains material non-public information (upcoming layoffs) with suggestion to act on it before public announcement

### Case 2: Ethical vs. Regulatory
**Message**: "Our investment returns aren't great, but don't mention the underperformance to clients."
**Classification**: ETHICAL_BREACH
**Reasoning**: Involves being dishonest with clients, but doesn't explicitly violate regulations about sharing non-public information

### Case 3: Confidential vs. Regulatory
**Message**: "I'm attaching our client database with all their financial information for your review."
**Classification**: CONFIDENTIAL_INFO
**Reasoning**: Shares confidential client data, but doesn't involve non-public information about securities or actions that would affect market prices

### Case 4: Subtle Harassment
**Message**: "Let's assign the technical project to John instead of Maria; women usually struggle with these complex systems."
**Classification**: HARASSMENT
**Reasoning**: Contains gender-based stereotyping that's inappropriate in a professional context

### Case 5: Ambiguous Case
**Message**: "Let's discuss the merger details offline instead of over email."
**Classification**: COMPLIANT
**Reasoning**: While suggesting to move sensitive discussion offline could be suspicious, it could also be a prudent security practice without more context indicating wrongdoing