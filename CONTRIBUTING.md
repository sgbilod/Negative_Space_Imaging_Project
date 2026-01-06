# Contributing Guidelines

Thank you for your interest in contributing to the Negative Space Imaging Project!

## Code of Conduct

We are committed to providing a welcoming and inclusive environment. Please:
- Be respectful and constructive
- Welcome diverse perspectives
- Focus on what's best for the project
- Report unacceptable behavior to maintainers

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/Negative_Space_Imaging_Project.git`
3. Create a feature branch: `git checkout -b feature/your-feature`
4. Follow the setup instructions in `DEVELOPMENT.md`

## Development Workflow

### Before Starting

- Check existing issues and PRs to avoid duplicate work
- Discuss significant changes in an issue first
- Follow the project's coding standards

### Making Changes

```bash
# Create feature branch
git checkout -b feature/your-feature

# Make changes following code standards
# Test your changes thoroughly
npm run test

# Ensure code quality
npm run lint
npm run format

# Commit with clear messages
git commit -m "fix: description of change

Optional longer description explaining why and how."
```

### Commit Messages

Follow conventional commits:

```
type(scope): subject

body (optional)
footer (optional, e.g. Fixes #123)
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Code style (formatting)
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance

**Examples:**
```
feat(auth): add JWT refresh token rotation
fix(api): handle null image uploads
docs(setup): clarify Docker installation
test(analyzer): add negative space detection tests
```

## Pull Request Process

### Before Submitting

1. **Update your branch:** `git pull origin main`
2. **Run tests:** `npm run test`
3. **Check linting:** `npm run lint`
4. **Test locally:** Verify all functionality works
5. **Update docs:** If behavior changed, update documentation

### Creating the PR

1. Use the provided PR template
2. Give a clear, descriptive title
3. Explain the problem and solution
4. Reference related issues: `Fixes #123`
5. Include testing instructions
6. Request reviewers

### PR Title Format

```
[type] Brief description

Examples:
[feat] Add batch image processing
[fix] Correct memory leak in analyzer
[docs] Update deployment guide
[test] Add integration tests for API
```

### During Review

- Respond to feedback professionally
- Make requested changes in new commits
- Re-request review after updates
- Be patient with the review process

## Code Standards

### TypeScript/Node.js

```typescript
// Use strict types
function analyze(image: Buffer): Promise<AnalysisResult> {
  // implementation
}

// Add error handling
try {
  const result = await process(data);
} catch (error) {
  logger.error('Processing failed:', error);
  throw new ApiError('Processing failed', 500);
}

// Use const/let, not var
const config = loadConfig();

// Use arrow functions
const map = (arr: number[]) => arr.map(x => x * 2);

// Document complex functions
/**
 * Analyzes image for negative space patterns
 * @param image - Image buffer to analyze
 * @param options - Configuration options
 * @returns Promise resolving to analysis results
 */
```

### Python

```python
# Use type hints
def analyze_image(image: np.ndarray) -> Dict[str, Any]:
    """Analyze image for negative space patterns."""
    # implementation
    pass

# Follow PEP 8
class ImageAnalyzer:
    """Main analyzer class."""

    def __init__(self, config: Dict) -> None:
        self.config = config

    def process(self, image: np.ndarray) -> Dict:
        """Process image and return results."""
        pass

# Use meaningful variable names
negative_space_regions = detect_regions(image)

# Add docstrings
def detect_regions(image: np.ndarray) -> List[Region]:
    """
    Detect negative space regions in image.

    Args:
        image: Input image array

    Returns:
        List of detected regions

    Raises:
        ValueError: If image format invalid
    """
```

### React Components

```typescript
// Use functional components
interface ImageViewerProps {
  imageUrl: string;
  onAnalyze?: (result: AnalysisResult) => void;
}

export const ImageViewer: React.FC<ImageViewerProps> = ({
  imageUrl,
  onAnalyze,
}) => {
  const [loading, setLoading] = useState(false);

  return (
    <div className="image-viewer">
      <img src={imageUrl} alt="Analysis target" />
    </div>
  );
};

// Use hooks for state
function useImageAnalysis(imageId: string) {
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<Error | null>(null);

  return { result, error };
}
```

## Testing Requirements

### Coverage Targets

- Overall: 70%+
- Critical paths: 90%+
- New features: 80%+

### Writing Tests

```typescript
describe('ImageAnalyzer', () => {
  it('should detect negative space regions', async () => {
    const analyzer = new ImageAnalyzer();
    const result = await analyzer.analyze(testImage);
    expect(result.regions).toHaveLength(3);
  });

  it('should handle invalid images', async () => {
    const analyzer = new ImageAnalyzer();
    await expect(analyzer.analyze(invalidImage)).rejects.toThrow();
  });
});
```

## Documentation

### What to Document

- Complex algorithms and their logic
- API endpoints with examples
- Configuration options
- Setup and deployment procedures
- Major architectural decisions

### Format

- Use Markdown for docs
- Include code examples
- Add diagrams where helpful
- Keep language clear and concise

## Performance

### Guidelines

- Profile before optimizing
- Focus on critical paths
- Cache expensive operations
- Use lazy loading where appropriate
- Monitor bundle size

### Tools

```bash
# Check bundle size
npm run build
npm run analyze

# Profile performance
npm run profile

# Load testing
npm run load-test
```

## Security

### Best Practices

- Never commit secrets or keys
- Use environment variables
- Validate all inputs
- Use HTTPS in production
- Keep dependencies updated
- Run security audits regularly

```bash
# Check for vulnerabilities
npm audit
npm audit fix

# Update dependencies
npm outdated
npm update
```

## CI/CD Pipeline

### Automated Checks

Every pull request and push to `main`/`develop` triggers automated checks via GitHub Actions:

#### Pipeline Stages

The unified CI/CD pipeline (`.github/workflows/build-deploy.yml`) executes:

1. **LINT** - Code quality, formatting, type checking
   - Python: black, flake8, isort, mypy
   - TypeScript: ESLint, Prettier, TypeScript compiler
   - YAML: yamllint

2. **TEST** - Unit & integration tests with coverage
   - Python: pytest with coverage reporting
   - Node.js: npm test
   - Coverage reports uploaded to Codecov
   - E2E smoke tests

3. **BUILD** - Docker image construction
   - Builds all Dockerfiles (API, Python, Frontend, Monitoring)
   - Generates SBOM (Software Bill of Materials)
   - Layer caching for performance

4. **SCAN** - Security vulnerability scanning
   - Trivy container image scanning
   - Dependency vulnerability checks
   - Secret detection
   - Reports to GitHub Security tab
   - Fails on CRITICAL/HIGH vulnerabilities

5. **PUSH** - Push to container registry
   - Only on successful scans
   - Only on `main`/`develop` branches
   - Tags: branch, commit SHA, semver
   - Pushes to ghcr.io

6. **DEPLOY** - Deployment to environments
   - Staging: Deployed from `develop`
   - Production: Deployed from `main`
   - Smoke tests run post-deployment
   - Monitored for issues

#### Workflow Status

Check the **Actions** tab for:
- Real-time build status
- Step-by-step execution logs
- Artifact downloads (test reports, SBOM)
- Failure details and logs

### Local Testing

Before pushing, run checks locally to catch issues early:

```bash
# Install development dependencies
pip install -r requirements.txt
npm install

# Run all linting
npm run lint
black --check sovereign quantum tests
flake8 sovereign quantum tests
mypy sovereign quantum

# Run tests with coverage
pytest --cov=sovereign --cov=quantum --cov-report=term-missing

# Build Docker images locally
docker build -f Dockerfile.api -t nsi-api:test .
docker build -f Dockerfile.python -t nsi-python:test .
docker build -f Dockerfile.frontend -t nsi-frontend:test .

# Scan images locally with Trivy
trivy image --severity CRITICAL,HIGH nsi-api:test
trivy image --severity CRITICAL,HIGH nsi-python:test
trivy image --severity CRITICAL,HIGH nsi-frontend:test

# Format code automatically
black sovereign quantum tests
isort --profile black sovereign quantum tests
npm run format
```

### Handling CI Failures

**Linting Failures:**
```bash
# Fix formatting
black sovereign quantum tests
isort --profile black sovereign quantum tests
npm run format

# Fix type errors
mypy sovereign quantum --fix-imports
```

**Test Failures:**
```bash
# Run tests locally to debug
pytest -xvs path/to/test_file.py::test_name

# View coverage reports
coverage html
open htmlcov/index.html
```

**Security Scan Failures:**
```bash
# Scan locally
trivy image nsi-api:test

# Update vulnerable dependencies
pip install --upgrade vulnerable-package
npm audit fix

# If false positive, add to .trivyignore
echo "CVE-2021-12345" >> .trivyignore
```

### Container Scanning Details

Container images are scanned for:
- **OS package vulnerabilities** - from base images
- **Python dependencies** - from requirements.txt
- **Node.js dependencies** - from package.json
- **Secrets** - AWS keys, API tokens, etc.

Scanning happens automatically in CI, but you can test locally:

```bash
# Install Trivy
brew install trivy  # macOS
sudo apt-get install trivy  # Ubuntu

# Scan local image
trivy image nsi-api:test

# Scan with config file
trivy image --config .github/trivy-config.yaml nsi-api:test

# Scan filesystem
trivy fs --severity CRITICAL,HIGH .

# Generate SARIF report
trivy image --format sarif nsi-api:test > report.sarif
```

### Branch Protection Rules

**Required checks for `main` branch:**
- Lint (must pass)
- Test (must pass)
- Build (must pass)
- Scan (must pass)
- At least 1 approval review

**Recommended:**
- Dismiss stale reviews when new commits pushed
- Require up-to-date branch before merging
- Include administrators in requirements

---

**Old Workflows**

Legacy workflows have been archived:
- `ci.yml.backup` - Previous CI configuration
- `ci-cd.yml.backup` - Previous CI/CD configuration

These are kept for reference but should not be used.

## Issue Management

### Reporting Bugs

Include:
- Clear description of problem
- Steps to reproduce
- Expected vs actual behavior
- Environment info (OS, Node version, etc.)
- Screenshots/logs if applicable

### Feature Requests

Include:
- Clear description of desired feature
- Use case and benefits
- Proposed solution (if any)
- Any relevant context

## Questions?

- Check existing issues and documentation
- Ask in GitHub Discussions
- Open an issue for clarification
- Contact maintainers

---

**Thank you for contributing!**

Your efforts help make this project better for everyone.

Last Updated: December 2025
