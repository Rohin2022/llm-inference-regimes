
# TASK: long prompt (PREFILL heavy)
SUMMARIZATION_PROMPT = """
Read the following government report carefully and produce a concise,
self-contained summary of its key findings, objectives, methods, and
conclusions. Focus on preserving the most important information.

Your summary should be substantially shorter than the original document,
while retaining enough detail to accurately represent the report.

REPORT:
{document}
"""


# TASK: short prompt (decode heavy)
REPORT_CREATION_PROMPT = """
Using the summary below as your source material, write a detailed,
self-contained government-style report expanding upon the information in the
summary. Explain the background, objectives, findings, evidence, implications,
and recommendations in depth. Produce a coherent and detailed report rather
than a brief summary.

SUMMARY:
{summary}

"""


def retrieve_full_prompt(task,text):
    if(task=="CREATION"):
        return REPORT_CREATION_PROMPT.format(text)
    elif(task=="SUMMARIZATION"):
        return SUMMARIZATION_PROMPT.format(text)
    
    return None