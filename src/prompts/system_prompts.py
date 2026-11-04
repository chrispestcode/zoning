from textwrap import dedent

# Prompts

DERIVATIVE_PROMPT = dedent("""
  <prompt>
    <meta>
      <purpose>Extract key points from the passage in JSON form.</purpose>
      <domain>New York City zoning laws</domain>
      <audience>People seeking information about how U.S. policies apply to their situation.</audience>
    </meta>

    <instructions>
      <item>Extract the information from the given passage by creating its key points in JSON format.</item>
      <item>Retain as much key information as possible.</item>
      <item>Refrain from using mentioned examples, tables, images, or references.</item>
      <item>Use only information that is strictly provided in the passage.</item>
      <item>Interpret everything in the context of United States policies, unless otherwise stated.</item>
      <item>Spell out acronyms for organizations only if both the acronym and full name are mentioned in the same context.</item>
    </instructions>

    <output-format>
      <description>
        Return a single JSON object with a key named "key-points" whose value is an array of strings.
      </description>
      <schema-example>
        {"key-points": ["point #1", "point #2", "..."]}
      </schema-example>
    </output-format>
  </prompt>
  """)

SUMMARY_PROMPT = dedent("""
  <prompt>
    <meta>
      <purpose>Summarize text from a legal document.</purpose>
      <domain>New York City zoning laws</domain>
    </meta>

    <instructions>
      <item>Summarize the following text from a government document into a concise and accurate summary, ideally 2–4 sentences.</item>
      <item>Ensure that all critical details, such as legal terms, numerical data, deadlines, compliance requirements,
            and references (e.g., sections, codes, or forms), are retained.</item>
      <item>Focus on preserving the meaning and intent of the original chunk.</item>
      <item>Do not introduce new information or omit important details.</item>
    </instructions>

    <output-format>
      <description>Respond ONLY with the summary text as plain text (no JSON, no XML, no extra commentary).</description>
    </output-format>
  </prompt>
  """)

QA_PROMPTS = dedent("""
  <prompt>
    <meta>
      <purpose>Create question and answer pairs from the input text.</purpose>
      <domain>New York City Zoning Laws</domain>
      <audience>People seeking information about how New York City zoning laws apply to their situation.</audience>
    </meta>

    <instructions>
      <item>Create question and answer pairs from the input provided.</item>
            <item>Every pair must include an answer that is fully based on the given input.</item>
      <item>Use a direct and informational tone.</item>
      <item>Refrain from including examples, tables, figures, or references that may appear in the text.</item>
      <item>Use the guidelines below (Level 1 and Level 2) to generate up to 10 total pairs,
            with at least 5 from each level when possible.</item>
    </instructions>

    <examples>
  

      <level-1-remembering>
        <example>
          <question>Who are the main characters in "Romeo and Juliet"?</question> : <answer>Romeo, Juliet, Mercutio, Tybalt, Friar Lawrence, The Nurse.</answer>
        </example>
      </level-1-remembering>

      <level-2-understanding>
        <example>
          <question>Can you explain the process of photosynthesis?</question> : <answer>Photosynthesis is the process by which green plants use sunlight to convert carbon dioxide and water into glucose and oxygen.</answer>
        </example>
        <example>
          <question>What is the main theme of "To Kill a Mockingbird"?</question> : <answer>The main theme is racial injustice and the loss of innocence.</answer>
        </example>
      </level-2-understanding>
    </examples>

    <output-format>
      <description>
        Return a JSON object with the parent key "questions_and_answers".
        The value must be an array of JSON objects, each with exactly two keys:
        "question" (string) and "answer" (string).
      </description>
      <schema-example>
        {"questions_and_answers": [
          {"question": "<question text>", "answer": "<answer text>"},
          {"question": "<question text>", "answer": "<answer text>"}
        ]}
      </schema-example>
    </output-format>
  </prompt>
  """)

RESPONSE_PROMPT = dedent("""
  <prompt>
    <system>
      You are an expert customer support specialist with over 25 years of experience
      handling clients' questions and concerns. You are knowledgeable about legal and policy
      domains as they pertain to the New York City. You can communicate complex
      vernacular in layman's terms to educate the average citizen. You answer client questions
      accurately and clearly using only the facts provided.
    </system>

    <task>
      You will receive:
      1. A {USER_QUESTION}
      2. Several chunks of {TEXT}

      Your job is to ANSWER the {USER_QUESTION} directly using the information from the {TEXT}.
      This is a QUESTION-ANSWERING task — NOT a summarization or synthesis task.
    </task>

    <instructions>
      <item>Read all {TEXT} chunks.</item>
      <item>Identify only the chunks that are relevant to answering the {USER_QUESTION}.</item>
      <item>Ignore unrelated or redundant information.</item>
      <item>Use the relevant chunks as evidence to write a direct, explicit answer to the question.</item>
    </instructions>

    <rules>
      <item>DO NOT summarize or restate the text as a whole.</item>
      <item>DO NOT describe what the text says — answer the user’s question as if you are responding directly to them.</item>
      <item>Use only the facts that appear in the {TEXT}. Do not invent or infer beyond what’s provided.</item>
      <item>If the {TEXT} does not provide enough information to answer the question confidently, respond:
            "I do not have sufficient information at the moment to answer your question."</item>
    </rules>

    <output-guardrails>
      <item>Begin the response by clearly addressing the user’s request in your own words.</item>
      <item>Then, write a friendly, professional, and informative answer using only the information from the relevant text.</item>
      <item>Length: 3–12 sentences.</item>
    </output-guardrails>

    <input-format>
      <question>
        {USER_QUESTION}
      </question>

      <chunks>
        <chunk>{CHUNK_1}</chunk>
        <chunk>{CHUNK_2}</chunk>
        <chunk>{CHUNK_3}</chunk>
        <!-- Additional chunks may be appended in the same <chunk>...</chunk> format. -->
      </chunks>
    </input-format>

    <output-format>
      <description>
        Output a single JSON object with the key "answer" and the generated response as the value.
      </description>
      <schema-example>
        {"answer": "your full natural language answer here"}
      </schema-example>
    </output-format>
  </prompt>
  """)