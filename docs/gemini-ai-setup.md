# Gemini AI Integration Setup Guide

This guide explains how to set up and configure Gemini AI analysis for the MCP Server Cheap LLM project.

## 🚀 Quick Setup

### 1. Get Gemini API Key

1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Copy the API key for use in GitHub secrets

### 2. Configure GitHub Repository

#### Required Secrets
Add these secrets to your GitHub repository (Settings → Secrets and variables → Actions):

- **`GEMINI_API_KEY`**: Your Gemini API key from Google AI Studio

#### Optional Variables
Add these variables to enable/disable AI features (Settings → Secrets and variables → Actions → Variables):

- **`ENABLE_AI_ANALYSIS`**: Set to `'true'` to enable AI analysis in CI Framework
- **`AI_ANALYSIS_LEVEL`**: Set to `'comprehensive'`, `'code-review'`, or `'basic'`

### 3. Verify Setup

Once configured, the AI analysis will automatically:

- **Trigger on Pull Requests** - Provide AI-powered code review
- **Trigger on CI Failures** - Analyze failures and suggest fixes
- **Available on Demand** - Manual comprehensive analysis via workflow dispatch

## 🤖 AI Analysis Features

### Automated Code Review

**Trigger**: Pull request opened, updated, or reopened

**Features**:
- Code quality assessment
- MCP protocol compliance check
- Performance optimization suggestions
- Security vulnerability detection
- Testing recommendations
- Documentation improvements

**Output**: Posted as PR comment with structured analysis

### CI Failure Analysis

**Trigger**: CI Framework workflow fails

**Features**:
- Root cause analysis of failures
- Specific fix suggestions with commands
- Prevention strategies
- Step-by-step debugging guide
- Priority assessment

**Output**: Creates GitHub issue with analysis and recommendations

### Manual Comprehensive Analysis

**Trigger**: Manual workflow dispatch

**Features**:
- Architecture assessment
- Code quality review
- Performance bottleneck identification
- Security assessment
- Strategic recommendations
- Prioritized action items

**Options**:
- **Analysis Type**: `code-review`, `security-analysis`, `performance-analysis`, `comprehensive`
- **Focus Area**: `all`, `src/cache`, `src/providers`, `src/core`, `tests`

## 🎯 AI Analysis Prompts

The AI analysis is tailored for the MCP Server project with specific context about:

### Project Context
- Python 3.12+ with async/await patterns
- MCP Protocol implementation
- Multi-provider LLM integration
- Zero-tolerance quality policy
- Performance and security requirements

### Analysis Focus Areas
1. **Code Quality** - Python best practices, type hints, error handling
2. **MCP Compliance** - Correct protocol implementation
3. **Performance** - Async patterns, caching, resource management
4. **Security** - Input validation, API key handling, vulnerabilities
5. **Testing** - Coverage gaps, test quality improvements
6. **Architecture** - Design patterns, maintainability

## 🔧 Configuration Files

### GEMINI.md
Project context file that provides:
- Architecture overview
- Technology stack details
- Quality standards
- Performance requirements
- Security considerations
- Common issues and solutions

### .change-patterns.toml
Includes AI configuration patterns:
```toml
# AI analysis configuration (require AI analysis)
ai_config = [
    "GEMINI.md",
    ".github/workflows/gemini-ai-analysis.yml"
]
```

### Workflow Files
- **`.github/workflows/gemini-ai-analysis.yml`** - Main AI analysis workflow
- **`.github/workflows/ci-framework.yml`** - Integration with CI Framework

## 💡 Usage Examples

### Manual Analysis Commands

**Comprehensive Analysis**:
```bash
gh workflow run gemini-ai-analysis.yml -f analysis_type=comprehensive -f focus_area=all
```

**Security Focus**:
```bash
gh workflow run gemini-ai-analysis.yml -f analysis_type=security-analysis -f focus_area=src/core
```

**Performance Analysis**:
```bash
gh workflow run gemini-ai-analysis.yml -f analysis_type=performance-analysis -f focus_area=src/cache
```

### Pull Request Interaction

The AI will automatically analyze pull requests and provide:
- Structured code review comments
- Specific suggestions with file:line references
- Overall assessment rating
- Action item recommendations

## 🔒 Security & Privacy

### Data Handling
- Code snippets sent to Gemini API for analysis only
- No persistent storage of code on Google servers
- API key securely stored in GitHub secrets
- Analysis results visible in GitHub interface only

### Best Practices
- Regularly rotate Gemini API keys
- Monitor API usage and costs
- Review AI suggestions before implementing
- Use AI analysis to complement, not replace, human review

## 🚨 Troubleshooting

### Common Issues

**AI Analysis Not Triggering**
1. Check `GEMINI_API_KEY` secret is set
2. Verify `ENABLE_AI_ANALYSIS` variable is `'true'`
3. Ensure proper workflow file permissions

**API Key Errors**
1. Verify API key validity in Google AI Studio
2. Check API key permissions and quotas
3. Regenerate API key if necessary

**Analysis Quality Issues**
1. Update `GEMINI.md` with more project context
2. Adjust analysis prompts in workflow file
3. Ensure recent codebase context in analysis

### Getting Help

- Check workflow run logs for error details
- Review Google AI Studio API quotas and usage
- Update `GEMINI.md` for better AI context
- File GitHub issues with `ai-analysis` label for support

## 📊 Benefits

### Development Quality
- **Consistent Reviews** - AI provides consistent code review standards
- **Knowledge Sharing** - AI explains complex patterns and best practices
- **Early Detection** - Catch issues before human review
- **Learning Tool** - Educational insights for junior developers

### CI/CD Efficiency
- **Faster Debugging** - AI analysis of CI failures saves investigation time
- **Proactive Insights** - Identify potential issues before they become problems
- **Documentation** - Automatic issue creation with detailed analysis

### Project Health
- **Architecture Guidance** - Strategic recommendations for codebase evolution
- **Performance Monitoring** - AI-powered performance bottleneck identification
- **Security Awareness** - Additional security review layer

The Gemini AI integration enhances the existing CI Framework without replacing human judgment, providing valuable insights to improve code quality and development efficiency.
