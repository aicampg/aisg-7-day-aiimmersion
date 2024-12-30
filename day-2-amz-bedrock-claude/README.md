# Day 2: Cloud AI with Amazon Bedrock & Claude

## Preface
Today we dive into cloud-based AI services, focusing on Amazon Bedrock and Claude. You'll learn how to set up and interact with professional-grade AI services, understanding their capabilities and best practices. This session bridges the gap between basic LLM interaction and production-ready AI implementation.

## Goals
By the end of today, you should:
- Have working access to Amazon Bedrock
- Understand Claude's capabilities
- Be able to make basic API calls
- Know how to handle responses
- Understand cloud AI best practices

## Main Resources
### Required Accounts
- AWS Account
- AWS Access Keys
- Claude API access
- Python environment(IDX)

### Core Services
- Amazon Bedrock Console
- Bedrock Runtime
- Claude API
- boto3 (AWS SDK for Python)

### Development Tools
- Python 3.8+
- Development environment from Day 1
- AWS CLI (optional)

## Tasks
1. AWS Setup
   - Configure AWS account
   - Request model access
   - Set up authentication
   - Test console access

2. Basic Integration
   - Install required packages
   - Configure credentials
   - Test basic calls
   - Handle responses

3. Claude Exploration
   - Understand capabilities
   - Test different prompts
   - Explore parameters
   - Document results

## Instructions
### AWS Configuration
1. Account Setup
   - Create AWS account if needed
   - Create AWS user
   - Add Read Bdrock permissions to user, and invoke LLM and stream
   - Create Access credentials(external service)
   - Note credentials safely

2. Model Management
   - Request access for Anthropic Sonnet
   - Fill out the request form


### Environment configuration
1. Environment Setup
   - Set up an IDX environment with jupyter notebook configured

2. Github code download
   - Make sure to 
   >git clone https://github.com/aicampg/aisg-5-day-aiimmersion.git
   - Alternatively go to https://idx.google.com/import?url=https://github.com/aicampg/aisg-7-day-aiimmersion
   - This will automatically import the repository
   - Open folder day-2 and select the workshop workbook
   - Run the workbook step by step, enter your credentials when asked for.

3. Create a virtual environment
   - Make sure your directory is your working directory
   - Create a new virtual environment
   > python -m venv your_environment
   - Activate the virtual environment
   > source lab_env/bin/activate
   - Now install requirements
   > pip install -r day-2-amz-bedrock-claude/requirements.txt

4. Run the jupyter notebook
   - If you are in IDX or in your personal workspace ignore the first cell for the pip install commands


## Further Resources
### Documentation
- AWS Bedrock: [https://docs.aws.amazon.com/bedrock/latest/userguide/what-is-bedrock.html](https://docs.aws.amazon.com/bedrock/latest/userguide/what-is-bedrock.html)
- Claude API: [Anthropic Bedrock documentation](https://docs.anthropic.com/en/api/claude-on-amazon-bedrock)

### Learning Materials
- [AWS Bedrock Getting Started](https://github.com/aicampg/aisg-5-day-aiimmersion/blob/main/day-2-amz-bedrock-claude/README.md)
- [Claude Prompt Design Guide](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview)
- [AWS SDK Tutorials](https://docs.aws.amazon.com/frauddetector/latest/ug/getting-started-python.html)

### Additional Reading
- [Cloud AI Best Practices](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/prompt-best-practices)
- [Security Guidelines](https://medium.com/@API4AI/api-security-best-practices-for-developers-cd7c0a2b6436)
- [Cost Management](https://cloud.google.com/transform/three-proven-strategies-for-optimizing-ai-costs)
- [Performance Optimization](https://www.techtarget.com/searchenterpriseai/tip/AI-model-optimization-How-to-do-it-and-why-it-matters)

### Troubleshooting
#### Authentication Issues
make sure you have done the following:
1. Makes sure the region matches in the following:
 - AWS user account for credentials 
 - Requested LLM Model API
 - The region in your client configuration
2. Make sure the AWS user account for API credentials has the right permissions:
 - General read Bedrock Read Permissions
 - Inline policy to invoke LLM inference models
#### API Response Errors
#### Rate Limits
#### Common Setup Problems

### Preparation for Day 3
- Review multi-modal concepts - [Google Documentation](https://cloud.google.com/use-cases/multimodal-ai)
- Check Google Cloud access - [GCP](https://cloud.google.com/)
- Understand vision APIs - [Documentation](https://ai.google.dev/gemini-api/docs/vision?lang=python)

### Tips
- Keep credentials secure
- Monitor API usage
- Test with small requests first
- Document error messages
- Use version control for code