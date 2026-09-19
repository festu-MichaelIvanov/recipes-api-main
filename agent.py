"""
Please provide the full URL to your recipes-api GitHub repository below.
"""

import asyncio
import os
from urllib.parse import urlparse

import dotenv
from github import Auth, Github

from llama_index.core.agent.workflow import (
    AgentOutput,
    ToolCall,
    ToolCallResult,
    AgentWorkflow,
    FunctionAgent,
)
from llama_index.core.workflow import Context
from llama_index.llms.openai import OpenAI
from llama_index.core.prompts import RichPromptTemplate


dotenv.load_dotenv()

github_token = os.getenv("GITHUB_TOKEN")
openai_api_key = os.getenv("OPENAI_API_KEY")

if not github_token:
    raise RuntimeError("GITHUB_TOKEN is not set")

if not openai_api_key:
    raise RuntimeError("OPENAI_API_KEY is not set")

git = Github(
    auth=Auth.Token(github_token)
)
repository = os.getenv("REPOSITORY")
pr_number = os.getenv("PR_NUMBER")

repo_url = f"https://github.com/{repository}.git"


def get_repo_name(url: str) -> str:
    path = urlparse(url).path.strip("/")

    if path.endswith(".git"):
        path = path[:-4]

    return path


repo = git.get_repo(get_repo_name(repo_url))


def get_pr_details(pr_number: int) -> dict:
    pull_request = repo.get_pull(pr_number)
    return {
        "author": pull_request.user.login,
        "title": pull_request.title,
        "body": pull_request.body or '',
        "diff_url": pull_request.diff_url,
        "state": pull_request.state,
        "head_sha": pull_request.head.sha,
        "commit_shas": [commit.sha for commit in pull_request.get_commits()]
    }


def get_commit_details(commit_sha: str) -> list[dict]:
    commit = repo.get_commit(commit_sha)
    return [
        {
            "filename": file.filename,
            "status": file.status,
            "additions": file.additions,
            "deletions": file.deletions,
            "changes": file.changes,
            "patch": file.patch,
        }
        for file in commit.files
    ]


def get_file_contents(
    file_path: str,
    ref: str | None = None,
) -> str:
    if ref is None:
        file = repo.get_contents(file_path)
    else:
        file = repo.get_contents(file_path, ref=ref)
    return file.decoded_content.decode("utf-8")


async def add_comment_to_state(ctx: Context, draft_comment: str) -> str:
    current_state = await ctx.store.get("state")
    current_state["draft_comment"] = draft_comment
    await ctx.store.set("state", current_state)
    return "Draft review saved."


async def add_context_to_state(ctx: Context, gathered_contexts: str) -> str:
    current_state = await ctx.store.get("state")
    current_state["gathered_contexts"] = gathered_contexts
    await ctx.store.set("state", current_state)
    return "PR context saved."


async def add_final_review_to_context(ctx: Context, final_review: str) -> str:
    current_state = await ctx.store.get("state")
    current_state["final_review"] = final_review
    await ctx.store.set("state", current_state)
    return "Final review saved."


def publish_final_review(pr_number: int, final_review_comment: str) -> None:
    pull_request = repo.get_pull(pr_number)
    pull_request.create_review(body=final_review_comment, event="COMMENT")


llm = OpenAI(
    model="gpt-5.6-sol",
    api_key=openai_api_key,
    reasoning_effort="none",
)

SYSTEM_PROMPT_FOR_CONTEXT_AGENT = """
You are ContextAgent.

Your ONLY job is to gather repository context requested by CommentorAgent.

When working with a pull request you MUST:

1. Call get_pr_details.
2. Read ALL commit_shas returned by get_pr_details.
3. Call get_commit_details for EVERY commit SHA.
4. Collect all changed files and patches.
5. Call get_file_contents if additional file contents are needed.
6. Call add_context_to_state with the complete gathered context.
7. IMMEDIATELY hand off to CommentorAgent.

IMPORTANT:
- You MUST NOT finish the workflow with a final response.
- You MUST NOT write a pull request review.
- You MUST always hand off to CommentorAgent after gathering context.
- If multiple commit SHAs exist, inspect ALL of them.
"""

SYSTEM_PROMPT_FOR_COMMENTOR_AGENT = """
You are CommentorAgent.

Your job is to write a complete pull request review.

Follow this workflow EXACTLY:

1. If the repository/PR context has not yet been gathered,
   hand off to ContextAgent.

2. Do NOT call add_comment_to_state just to request information.
   add_comment_to_state is ONLY for saving the COMPLETE final draft review.

3. After ContextAgent returns with the required context, write a
   200-300 word Markdown review.

The review must include:
- What is good about the PR.
- Whether contribution rules were followed and what is missing.
- Whether tests exist for new functionality.
- Whether migrations exist for new models.
- Whether new endpoints are documented, when applicable.
- Specific changed lines quoted with improvement suggestions.
- Directly address the author.

4. Call add_comment_to_state with the COMPLETE review text.

5. After add_comment_to_state succeeds, IMMEDIATELY hand off to
   ReviewAndPostingAgent.

IMPORTANT:
- Do NOT finish the workflow with a final response.
- Do NOT save requests for context as draft_comment.
- Do NOT hand off to ContextAgent after the complete review has been saved,
  unless ReviewAndPostingAgent explicitly requests missing information.
"""

SYSTEM_PROMPT_FOR_REVIEW_AGENT = """
You are ReviewAndPostingAgent.

You coordinate and finalize the pull request review.

Follow this workflow EXACTLY:

1. If no review has been drafted yet, hand off to CommentorAgent.

2. When CommentorAgent hands control back to you, inspect the existing
   draft review from the conversation history.

3. Verify that it:
   - is approximately 200-300 words in Markdown;
   - explains what is good about the PR;
   - discusses contribution requirements;
   - discusses tests;
   - discusses migrations for new models;
   - discusses endpoint documentation when applicable;
   - quotes changed lines and gives improvement suggestions.

4. If the draft is insufficient, hand off to CommentorAgent with exact
   instructions about what to fix.

5. If the draft is satisfactory:
   - call add_final_review_to_context with the complete review;
   - call publish_final_review with the PR number and the complete review.

6. Only finish after publish_final_review succeeds.
"""


context_agent = FunctionAgent(
    name="ContextAgent",
    description="Gathers PR details, changed files, diffs and repository context.",
    system_prompt=SYSTEM_PROMPT_FOR_CONTEXT_AGENT,
    llm=llm,
    tools=[
        get_pr_details,
        get_commit_details,
        get_file_contents,
        add_context_to_state,
    ],
    can_handoff_to=["CommentorAgent"],
)

commentor_agent = FunctionAgent(
    llm=llm,
    name="CommentorAgent",
    description="Creates a complete pull request review from gathered context.",
    system_prompt=SYSTEM_PROMPT_FOR_COMMENTOR_AGENT,
    tools=[add_comment_to_state],
    can_handoff_to=["ContextAgent", "ReviewAndPostingAgent"],
)

review_and_posting_agent = FunctionAgent(
    llm=llm,
    name="ReviewAndPostingAgent",
    description="Validates and publishes the completed pull request review.",
    system_prompt=SYSTEM_PROMPT_FOR_REVIEW_AGENT,
    tools=[add_final_review_to_context, publish_final_review],
)

workflow_agent = AgentWorkflow(
    agents=[context_agent, commentor_agent, review_and_posting_agent],
    root_agent=review_and_posting_agent.name,
    initial_state={
        "gathered_contexts": "",
        "draft_comment": "",
        "final_review": "",
    },
)



async def main():
    prompt = RichPromptTemplate(f"Write review for PR {pr_number}")

    handler = workflow_agent.run(prompt.format())

    current_agent = None
    async for event in handler.stream_events():
        if hasattr(event, "current_agent_name") and event.current_agent_name != current_agent:
            current_agent = event.current_agent_name
            print(f"Current agent: {current_agent}")
        elif isinstance(event, AgentOutput):
            if event.response.content:
                print("\\n\\nFinal response:", event.response.content)
            if event.tool_calls:
                print("Selected tools: ", [call.tool_name for call in event.tool_calls])
        elif isinstance(event, ToolCallResult):
            print(f"Output from tool: {event.tool_output}")
        elif isinstance(event, ToolCall):
            print(f"Calling selected tool: {event.tool_name}, with arguments: {event.tool_kwargs}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    finally:
        git.close()
